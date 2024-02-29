import argparse
import os
import functools
import glob
import re
import subprocess
from copy import deepcopy

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import textgrid
import torch
from face_detector import FaceDetector
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Subset

NUM_CLASSES = 13
VISEME_2_PHONEME = {  # lee and york 2002
    'p': ['p', 'b', 'm', 'em'],
    'f': ['f', 'v'],
    't': ['t', 'd', 's', 'z', 'th', 'dh', 'dx'],
    'w': ['w', 'wh', 'r'],
    'ch': ['ch', 'jh', 'sh', 'zh'],
    'ey': ['eh', 'ey', 'ae', 'aw'],
    'k': ['k', 'g', 'n', 'l', 'nx', 'hh', 'y', 'el', 'en', 'ng'],
    'iy': ['iy', 'ih'],
    'aa': ['aa', 'aa1'],
    'ah': ['ah', 'ax', 'ay'],
    'er': ['er'],
    'ao': ['ao', 'oy', 'ix', 'ow'],
    'uh': ['uh', 'uw'],
}
PHONEME_REGEX = r'([A-Z]+)\d*'
PHONEME_2_VISEME = {
    phoneme.upper(): viseme.upper()
    for viseme, phonemes in VISEME_2_PHONEME.items()
    for phoneme in phonemes
}


class BaseDataset:

    def __init__(self, main_directory, df, video_format, **kwargs):
        self.main_directory = main_directory
        self.setup_directory('')
        self.video_directory = self.setup_directory('video')
        self.audio_directory = self.setup_directory('audio')
        self.processed_data_directory = self.setup_directory('processed_data')
        self.phoneme_dataset_path = \
            os.path.join(self.main_directory, 'phoneme_dataset.csv')
        self.processed_roi_directory = self.setup_directory('processed_roi')
        self.preprocessed_dataset_path = \
            os.path.join(self.main_directory, 'preprocessed_dataset.csv')
        self.video_format = video_format
        self.df = df if df is not None else self.preprocess_csv()
        self.transforms = kwargs.get('transforms')

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    def __getitem__(self, index):
        row = self.df.iloc[index]
        viseme = row['viseme']

        mouth_region = cv2.imread(row['image_path'])
        target = self.class_id_mapper[viseme]

        mouth_region = Image.fromarray(mouth_region)
        target = torch.tensor(target, dtype=torch.long)

        mouth_region = self.transforms(mouth_region)

        return mouth_region, target

    @property
    @functools.lru_cache()
    def speakers(self):
        return list(self.df['speaker'].unique())

    @property
    @functools.lru_cache()
    def viseme_classes(self):
        return list(self.df['viseme'].unique())

    @property
    def class_id_mapper(self):
        return {
            viseme: _id
            for _id, viseme in enumerate(self.viseme_classes)
        }

    def extend(self, df):
        self.df = pd.concat([self.df, df]).reset_index(drop=True)

        return self

    def setup_directory(self, name):
        path = os.path.join(self.main_directory, name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def get_num_classes(self):
        return len(self.viseme_classes)

    def generate_speaker_subsets(self, train_speakers):
        """split dataset based on speaker ids"""
        df = self.df

        # train_indices = df.index[df['speaker'].isin(train_speakers)]
        # val_indices = df.index[-df['speaker'].isin(train_speakers)]

        train_df = df[df['speaker'].isin(train_speakers)].reset_index(drop=True)
        val_df = df[-df['speaker'].isin(train_speakers)].reset_index(drop=True)

        train_copy = deepcopy(self)
        train_copy.df = train_df

        test_copy = deepcopy(self)
        test_copy.df = val_df

        return Subset(train_copy, train_df.index), \
               Subset(test_copy, val_df.index)

    def get_class_counts(self):
        class_counts = {}
        for index, row in self.df.iterrows():
            viseme = row['viseme']
            class_counts[viseme] = class_counts.get(viseme, 0) + 1

        return class_counts

    def show_class_balance(self):
        class_counts = self.get_class_counts()

        x = list(class_counts.keys())
        y = list(class_counts.values())
        plt.bar(x, y)
        plt.xlabel('Viseme')
        plt.ylabel('Count')
        plt.show()

    def show_class_examples(self):
        df = self.df
        print(self.viseme_classes)
        for viseme_class in self.viseme_classes:
            # randomly shuffle dataset
            df = df.sample(frac=1)

            viseme_examples = df[df['viseme'] == viseme_class].head(25)
            images = [cv2.imread(row['image_path'])
                      for index, row in viseme_examples.iterrows()]

            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(f'Viseme Examples: {viseme_class}')
            columns = 5
            rows = 5
            for i in range(columns * rows):
                image = images[i]
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(image)

            plt.show()

    def get_valid_clips_percentage(self):
        """valid clips are phoneme videos that exist with frames"""
        total_num_phoneme_clips = len(self.df)
        num_valid_phoneme_clips = 0
        num_exist_phoneme_clips = 0
        for index, row in self.df.iterrows():
            video_path = row['video path']
            if os.path.exists(video_path):
                num_exist_phoneme_clips += 1
                frames = self.get_frames(video_path)
                if frames:
                    num_valid_phoneme_clips += 1

        percentage_valid = num_valid_phoneme_clips / total_num_phoneme_clips
        print(f'Total in df: {total_num_phoneme_clips}, '
              f'Total exist: {num_exist_phoneme_clips}, '
              f'Total exist with frames: {num_valid_phoneme_clips}, '
              f'Percentage: {percentage_valid}')

    def convert_to_viseme(self, phoneme):
        # remove digits from end of phoneme
        phoneme = re.match(PHONEME_REGEX, phoneme).groups()[0]

        return PHONEME_2_VISEME.get(phoneme.upper())

    def get_num_frames(self, video_path):
        video_reader = cv2.VideoCapture(video_path)
        num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        video_reader.release()

        return num_frames

    def get_frames(self, video_path):
        video_reader = cv2.VideoCapture(video_path)
        frames = []
        while True:
            success, frame = video_reader.read()
            if not success:
                break
            frames.append(frame)
        video_reader.release()

        return frames

    def get_center_frame(self, video_path):
        frames = self.get_frames(video_path)
        if frames:
            center_index = len(frames) // 2
            return frames[center_index]

        return None

    def preprocess_csv(self):
        if not os.path.exists(self.preprocessed_dataset_path):
            if not os.path.exists(self.phoneme_dataset_path):
                return None

            df = pd.read_csv(self.phoneme_dataset_path,
                             names=['speaker', 'video_path', 'phoneme'])

            # remove videos with no phonemes
            df = df.dropna()

            # drop sils
            df = df[df['phoneme'] != 'sil']

            unique_phonemes = df['phoneme'].unique()
            for phoneme in unique_phonemes:
                if not self.convert_to_viseme(phoneme):
                    print(phoneme, 'not in mapping')
                    exit()

            # convert phonemes to visemes
            df['viseme'] = df.apply(
                lambda row: self.convert_to_viseme(row['phoneme']),
                axis=1
            )

            # drop SPs
            df = df[df['viseme'] != 'SP']

            # drop any bad videos (no frames or faces detected in them)
            face_detector = FaceDetector('mouth')
            df_copy = df.copy()
            for index, row in tqdm(df.iterrows()):
                video_path = row['video_path']
                image_path = os.path.join(
                    self.processed_roi_directory,
                    os.path.basename(video_path).replace('.mp4', '.jpg')
                )
                if os.path.exists(image_path):
                    df_copy.loc[index, 'image_path'] = image_path
                    continue

                center_frame = self.get_center_frame(video_path)
                if center_frame is None:
                    # delete video path
                    df_copy = df_copy[df_copy.index != index]
                else:
                    mouth_region = face_detector.extract_roi(center_frame)[0]
                    if mouth_region is None:
                        # delete video path
                        df_copy = df_copy[df_copy.index != index]
                    else:
                        try:
                            cv2.imwrite(image_path, mouth_region)
                        except cv2.error:
                            # write error
                            print('Write error')
                            df_copy = df_copy[df_copy.index != index]
                            continue

                        # add the image path to the df - speeds up __getitem__
                        df_copy.loc[index, 'image_path'] = image_path

            # reset indices
            df = df_copy.reset_index(drop=True)

            # save for later use
            df.to_csv(self.preprocessed_dataset_path)
        else:
            df = pd.read_csv(self.preprocessed_dataset_path)

        return df

    def extract_phoneme_video_clips(self):
        """Extract phoneme clips from phoneme alignment textgrids using
        ffmpeg"""
        text_grid_paths = glob.glob(os.path.join(self.processed_data_directory,
                                                 '*.TextGrid'))

        for text_grid_path in tqdm(text_grid_paths):
            video_name = os.path.basename(text_grid_path).replace(
                'TextGrid',
                self.video_format)
            video_path = os.path.join(self.processed_data_directory,
                                      video_name)
            tg = textgrid.TextGrid.fromFile(text_grid_path)
            phonemes = tg[0]
            for i, phoneme in enumerate(phonemes):
                # print(phoneme.minTime, phoneme.maxTime, phoneme.mark)
                output_video_path = os.path.join(
                    self.processed_data_directory,
                    video_name.replace(f'.{self.video_format}',
                                       f'_{i+1}.mp4')
                )
                if os.path.exists(output_video_path):
                    continue

                # TODO:
                #  ffmpeg -y -an -i {video_path} -ss 00:00:00.790
                #  -to 00:00:01.030 {viseme_video_path}
                #  -an = no audio

                # command = [
                #     'ffmpeg', '-y',
                #     '-i', f'{video_path}',
                #     '-force_key_frames',
                #     f'{float(phoneme.minTime):.2f},{float(phoneme.maxTime):.2f}',
                #     'temp.mp4'
                # ]
                # subprocess.call(command)

                # command = [
                #     'ffmpeg', '-y',
                #     '-ss', f'{float(phoneme.minTime):.2f}',
                #     '-i', 'temp.mp4',
                #     '-t',
                #     f'{float(phoneme.maxTime) - float(phoneme.minTime):.2f}',
                #     f'{output_video_path}'
                # ]
                # # print(command)
                # subprocess.call(command)  # this is blocking

                command = [
                    'ffmpeg', '-y',
                    '-loglevel', 'error',
                    '-i', f'{video_path}',
                    '-ss', f'{float(phoneme.minTime):.2f}',
                    '-to', f'{float(phoneme.maxTime):.2f}',
                    f'{output_video_path}'
                ]
                subprocess.call(command)  # blocking

                speaker_id = video_name.split('_')[0]
                with open(self.phoneme_dataset_path, 'a') as f:
                    f.write(
                        f'{speaker_id},{output_video_path},{phoneme.mark}\n'
                    )

    def forced_alignment_prepare(self):
        raise NotImplementedError

    def get_next_file_triple(self):
        raise NotImplementedError


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_directory')

    sub_parsers = parser.add_subparsers(dest='run_type')

    for run_type in [
        'forced_alignment_prepare',
        'extract_phoneme_video_clips',
        'get_valid_clips_percentage',
        'show_class_balance',
        'preprocess_csv',
        'show_class_examples'
    ]:
        sub_parsers.add_parser(run_type)

    return parser, sub_parsers
