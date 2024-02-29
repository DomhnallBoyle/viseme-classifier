import argparse
import os
import random
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import GridDataset, Wav2LipDataset
from models import Inception, Nvidia
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 4  # rule of thumb: 4 * NUM_GPU
MAX_STOPPING_PATIENCE = 1000
SEED = 2020

# TODO: Methods to try:
#  Copy Alex Model (smaller model)
#  CNN + LSTM
#  Optical Flow + CNN


def train(args):
    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # creating the 2 datasets
    grid_dataset = GridDataset(args.grid_path)
    wav2lip_dataset = Wav2LipDataset(args.wav2lip_path)
    dataset = grid_dataset.extend(wav2lip_dataset.df)
    if args.debug:
        dataset.show_class_balance()
        dataset.show_class_examples()

    wav2lip_speakers = wav2lip_dataset.speakers
    print('Wav2Lip Speakers:', wav2lip_speakers)

    # TODO:
    #  Weight Decay for Adam e.g. 0.001
    #  LR scheduler
    cnns = {
        'inception': Inception,
        'nvidia': Nvidia
    }
    print(args.__dict__)
    model = cnns[args.cnn](num_classes=dataset.get_num_classes(),
                           **args.__dict__).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # split dataset by speaker ids
    # ensure more samples of visemes classes for training
    speakers = dataset.speakers
    print('All Speakers:', speakers)
    speaker_split = int(len(speakers) * 0.7)
    while True:
        random.shuffle(speakers)
        train_speakers = speakers[:speaker_split]

        # # ensure synthetic speakers in training
        # if not all([speaker in train_speakers
        #             for speaker in wav2lip_speakers]):
        #     continue

        train_set, validation_set = \
            dataset.generate_speaker_subsets(train_speakers)

        training_class_counts = train_set.dataset.get_class_counts()
        val_class_counts = validation_set.dataset.get_class_counts()

        if len(training_class_counts) == len(val_class_counts) \
                == dataset.get_num_classes() and \
                all(training_class_counts[viseme] > val_class_counts[viseme]
                    for viseme in training_class_counts.keys()):
            break

    print('Train Speakers:', train_speakers)
    print('Validation Speakers:', list(set(speakers) - set(train_speakers)))

    # TODO: Theory: training transforms don't help validation
    # set our augmentation transforms for train and validation sets
    train_set.dataset.transforms = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.CenterCrop(model.input_size),
        transforms.RandomHorizontalFlip(),  # p = 0.5
        transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
        #                        hue=0.5),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    validation_set.dataset.transforms = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.CenterCrop(model.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = train_set.dataset
    training_class_counts = train_dataset.get_class_counts()
    training_class_sample_counts = [
        training_class_counts[viseme]
        for viseme in train_dataset.viseme_classes
    ]

    if args.imbalanced_over_sampling:
        # apply over-sampling of imbalanced classes to training set
        weights = 1. / torch.tensor(training_class_sample_counts,
                                    dtype=torch.float)
        sample_weights = weights[[
            train_dataset.viseme_classes.index(row['viseme'])
            for index, row in train_dataset.df.iterrows()
        ]]
        assert len(sample_weights) == len(train_set)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    else:
        # to balance classes, use minimum class count
        df = train_set.dataset.df
        np.random.shuffle(df.values)
        df = df.groupby('viseme').head(min(training_class_sample_counts))
        df.reset_index(inplace=True)
        train_set.dataset.df = df
        train_set.indices = df.index
        sampler = None

    print('Training class counts:', train_set.dataset.get_class_counts())
    print('Validation class counts:',
          validation_set.dataset.get_class_counts())

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              sampler=sampler,
                              pin_memory=True)

    # no sampler given to validation set, no need to shuffle either
    validation_loader = DataLoader(dataset=validation_set,
                                   batch_size=BATCH_SIZE,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=True)

    def show_img(img):
        plt.figure(figsize=(18, 15))
        # unnormalize
        img = img / 2 + 0.5
        npimg = img.numpy()
        npimg = np.clip(npimg, 0., 1.)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def evaluate(_outputs, _actual):
        # https://discuss.pytorch.org/t/cross-entropy-loss-get-predicted-class/58215/9
        # returns indices of max values
        _, predictions = torch.max(_outputs.data, 1)

        num_correct = (predictions == _actual).sum()
        num_samples = predictions.size(0)

        loss = loss_function(_outputs, _actual)
        accuracy = round(float(num_correct) / float(num_samples), 2)

        return loss, accuracy, predictions

    def run_validation(_model, _validation_loader):
        av_loss, av_accuracy = 0, 0
        _model.eval()  # get model ready for eval e.g. no dropout
        _labels_d = {}
        accumulated_actual, accumulated_preds = torch.LongTensor(),\
                                                torch.LongTensor()

        with torch.no_grad():
            for x, y in _validation_loader:
                x = x.to(device)
                y = y.to(device)

                _labels_d = cumulate_batch_labels(y, _labels_d)

                _outputs = _model(x)

                _val_loss, _val_accuracy, _outputs = evaluate(_outputs, y)
                av_loss += _val_loss.item()
                av_accuracy += _val_accuracy
                accumulated_actual = torch.cat((accumulated_actual, y.cpu()))
                accumulated_preds = torch.cat((accumulated_preds, _outputs.cpu()))

        av_loss /= len(_validation_loader)
        av_accuracy /= len(_validation_loader)

        print('Validation class counts:', _labels_d)

        # show confusion matrix and classification report
        class_id_mapper = dataset.class_id_mapper
        ids_to_class = {v: k for k, v in class_id_mapper.items()}
        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(accumulated_actual, accumulated_preds)
        ).rename(columns=ids_to_class, index=ids_to_class)
        sns.heatmap(confusion_matrix_df, annot=True)
        print(classification_report(accumulated_actual, accumulated_preds))
        plt.draw()

        return av_loss, av_accuracy

    def cumulate_batch_labels(_labels, _labels_d):
        for label in _labels:
            label = label.item()
            _labels_d[label] = _labels_d.get(label, 0) + 1

        return _labels_d

    min_val_loss = float('inf')
    stopping_patience = 0
    for epoch in range(1, args.epochs+1):
        model.train()

        av_train_loss, av_train_accuracy = 0, 0
        loop = tqdm(train_loader, total=len(train_loader), unit='batch')
        labels_d = {}
        for i, (frames, labels) in enumerate(loop):
            # there are len(dataset) / BATCH_SIZE iterations (batches)
            # from this iterator

            # classes should hopefully be balanced at the end of each epoch
            # function to check
            labels_d = cumulate_batch_labels(labels, labels_d)

            # show images
            if args.debug:
                show_img(torchvision.utils.make_grid(frames))

            # each iteration contains BATCH_SIZE frames
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # zero the param gradients

            outputs = model(frames)  # contains BATCH_SIZE outputs

            train_loss, train_accuracy = evaluate(outputs, labels)[:2]
            av_train_loss += train_loss.item()
            av_train_accuracy += train_accuracy

            train_loss.backward()  # back propagation of loss
            optimizer.step()  # gradient descent step

            # tqdm stuff
            loop.set_description(f'Epoch {epoch}/{args.epochs}')
            loop.set_postfix(train_loss=train_loss.item(),
                             train_acc=train_accuracy)

        print('Training class counts:', labels_d)

        av_train_loss /= len(train_loader)
        av_train_accuracy /= len(train_loader)
        av_val_loss, av_val_accuracy = run_validation(model, validation_loader)
        print(f'Training Loss: {av_train_loss}, Accuracy: {av_train_accuracy}')
        print(f'Validation Loss: {av_val_loss}, Accuracy: {av_val_accuracy}')

        # save training stats
        with open(f'training_{start_time}.csv', 'a') as f:
            f.write(f'{epoch},'
                    f'{args.epochs},'
                    f'{av_train_loss},'
                    f'{av_train_accuracy},'
                    f'{av_val_loss},'
                    f'{av_val_accuracy}\n')

        # early stopping
        if av_val_loss < min_val_loss:
            print('Validation loss improved...', end='')
            min_val_loss = av_val_loss
            stopping_patience = 0
            # save the model after every validation improvement
            torch.save(model.state_dict(),
                       os.path.join(CURRENT_DIRECTORY,
                                    f'model_epoch_{epoch}.pkl'))
        else:
            # if loss doesn't keep decreasing
            if stopping_patience == args.max_stopping_patience:
                print('Early stopping...exiting')
                break
            else:
                stopping_patience += 1
                print('Validation loss not improved...', end='')
        print('stopping patience:', stopping_patience)

    plt.show()


def train_analysis(args):
    """plot training graphs"""

    fig, (ax1, ax2) = plt.subplots(2)

    def animate(i):
        df = pd.read_csv(args.file_path,
                         names=['epoch', 'max_epochs', 'train_loss',
                                'train_acc',
                                'val_loss', 'val_acc'])

        ax1.clear()
        ax2.clear()

        x = df['epoch']
        acc_ys, loss_ys = [df['train_acc'], df['val_acc']], \
                          [df['train_loss'], df['val_loss']]
        labels = ['train', 'val']
        max_epochs = df['max_epochs'].unique()[0]

        # accuracy graphs
        for y, label in zip(acc_ys, labels):
            ax1.plot(x, y, label=label)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylim((0, 1.01))
        ax1.set_xlim((1, max_epochs))
        ax1.legend()

        # loss graphs
        max_loss = max(df['train_loss'].max(), df['val_loss'].max())
        for y, label in zip(loss_ys, labels):
            ax2.plot(x, y, label=label)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim((0, max_loss+1))
        ax2.set_xlim((1, max_epochs))
        ax2.legend()

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


def main(args):
    f = {
        'train': train,
        'train_analysis': train_analysis
    }

    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('train')
    parser_1.add_argument('grid_path')
    parser_1.add_argument('wav2lip_path')
    parser_1.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser_1.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser_1.add_argument('--max_stopping_patience', type=int,
                          default=MAX_STOPPING_PATIENCE)
    parser_1.add_argument('--seed', type=int, default=SEED)
    parser_1.add_argument('--debug', action='store_true')
    parser_1.add_argument('--imbalanced_over_sampling', action='store_true')
    parser_1.add_argument('--cnn', choices=['inception', 'nvidia'],
                          default='inception')
    parser_1.add_argument('--freeze_layers', action='store_true')

    parser_2 = sub_parsers.add_parser('train_analysis')
    parser_2.add_argument('file_path')

    main(parser.parse_args())
