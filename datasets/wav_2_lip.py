import random
import shutil

from datasets.base import *
from tqdm import tqdm
from torch.utils.data import Dataset

PROSODY_ENG_DICT = {}
with open('eng.dict', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        split_line = line.split(' ')
        word, phonemes = split_line[0], split_line[1:]
        PROSODY_ENG_DICT[word.lower()] = phonemes


class Wav2LipDataset(BaseDataset, Dataset):
    """Steps:
        1) generate synthetic phrases
        2) use wav2lip to generate the synthetic videos (requires video and
        phrases)
        3) use forced alignment (requires audio and phrases)
    """

    def __init__(self, main_directory, min_visemes=100,
                 max_words_per_phrase=10, df=None, **kwargs):
        super().__init__(main_directory, df, video_format='mp4', **kwargs)
        self.phrases_path = os.path.join(main_directory,
                                         'synthetic_phrases.txt')
        self.phrases_directory = self.setup_directory('phrases')
        self.min_visemes = min_visemes
        self.max_words_per_phrase = max_words_per_phrase

    def get_next_file_triple(self):
        with open(self.phrases_path, 'r') as f:
            phrases = f.read().splitlines()

        phrase_to_filename = lambda phrase: \
            re.sub(r'[\?\!]', '',
                   str(phrase).lower().strip().replace(' ', '_'))

        for phrase in phrases:
            filename = phrase_to_filename(phrase)
            video_paths = glob.glob(os.path.join(
                self.video_directory,
                f'{filename}_*.mp4'
            ))
            phrase_path = os.path.join(self.phrases_directory,
                                       f'{filename}.txt')
            if not os.path.exists(phrase_path):
                with open(phrase_path, 'w') as f:
                    f.write(phrase.upper())
            audio_path = os.path.join(self.audio_directory, f'{filename}.wav')
            assert os.path.exists(audio_path)
            for video_path in video_paths:
                assert os.path.exists(video_path)
                yield phrase_path, video_path, audio_path

    def forced_alignment_prepare(self):
        for paths in tqdm(self.get_next_file_triple()):
            if any([path is None for path in paths]):
                continue

            phrase_path, video_path, audio_path = paths
            speaker_id = video_path.split('_')[-1].replace('.mp4', '')
            phrase_filename = \
                '_'.join(os.path.basename(video_path).split('_')[:-1])

            new_lab_path = os.path.join(
                self.processed_data_directory,
                f'{speaker_id}_{phrase_filename}.lab'
            )
            new_audio_path = os.path.join(
                self.processed_data_directory,
                f'{speaker_id}_{phrase_filename}.wav'
            )
            new_video_path = os.path.join(
                self.processed_data_directory,
                f'{speaker_id}_{phrase_filename}.mp4'
            )

            if all([os.path.exists(new_path)
                    for new_path in [new_lab_path, new_audio_path,
                                     new_video_path]]):
                continue

            # copy files to the data directory
            for old_path, new_path in zip(
                [phrase_path, audio_path, video_path],
                [new_lab_path, new_audio_path, new_video_path]
            ):
                shutil.copy(old_path, new_path)

    def generate_synthetic_phrases(self):
        """Try to generate a synthetic phrases dataset based on balanced
        visemes
        """
        words_to_visemes = {}
        dict_viseme_count = {k: 0 for k in VISEME_2_PHONEME.keys()}
        with open('cmudict-en-us.dict', 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                split_line = line.split(' ')
                word, phonemes = split_line[0], split_line[1:]
                if '(' in word or not PROSODY_ENG_DICT.get(word):
                    continue
                visemes = [PHONEME_2_VISEME[phoneme] for phoneme in phonemes]
                words_to_visemes[word] = visemes
                for viseme in visemes:
                    dict_viseme_count[viseme.lower()] += 1

        print('Min dict viseme count:',
              min(dict_viseme_count.items(), key=lambda x: x[1]))

        all_words = list(words_to_visemes.keys())

        # find smallest number of phrases that construct the least set amount
        # of visemes you desire
        min_num_phrases = 1000000
        smallest_phrase_set, best_viseme_count = None, None
        while True:
            try:
                phrases = []
                viseme_counts = {k: 0 for k in VISEME_2_PHONEME.keys()}
                while True:
                    num_words = random.randint(2, self.max_words_per_phrase)
                    phrase = [random.choice(all_words) for _ in
                              range(num_words)]
                    if any(['\'' in word for word in phrase]):
                        # ignore phrases with contractions
                        continue
                    phrase_visemes = [viseme.lower() for word in phrase
                                      for viseme in words_to_visemes[word]]
                    for viseme in phrase_visemes:
                        viseme_counts[viseme] += 1
                    phrases.append(' '.join(phrase))

                    if all([num_visemes >= self.min_visemes
                            for num_visemes in viseme_counts.values()]):
                        break

                if len(phrases) < min_num_phrases:
                    min_num_phrases = len(phrases)
                    smallest_phrase_set = phrases
                    best_viseme_count = viseme_counts
            except KeyboardInterrupt:
                break

        print(len(smallest_phrase_set), best_viseme_count)
        with open(self.phrases_path, 'w') as f:
            for phrase in smallest_phrase_set:
                f.write(f'{phrase}\n')


def main(args):
    dataset = Wav2LipDataset(**args.__dict__)
    getattr(dataset, args.run_type)()


if __name__ == '__main__':
    parser, sub_parsers = build_parser()

    parser_1 = sub_parsers.add_parser('generate_synthetic_phrases')
    parser_1.add_argument('--min_visemes', type=int, default=100)
    parser_1.add_argument('--max_words_per_phrase', type=int, default=10)

    main(parser.parse_args())
