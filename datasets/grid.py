import shutil

from datasets.base import *
from tqdm import tqdm
from torch.utils.data import Dataset

# NOTE: Grid audio not same length as video
# This breaks the extraction of visemes using forced alignment


class GridDataset(BaseDataset, Dataset):

    def __init__(self, main_directory, df=None, **kwargs):
        super().__init__(main_directory, df, video_format='mpg', **kwargs)
        self.align_directory = os.path.join(self.main_directory, 'align')

    def get_next_file_triple(self):
        get_sub_directories = lambda directory: [
            os.path.join(directory, dir_name)
            for dir_name in os.listdir(directory)
        ]

        def find_equivalent_file(directory, extension):
            for sub_directory in get_sub_directories(directory):
                for file_path in glob.glob(os.path.join(sub_directory,
                                                        f'*.{extension}')):
                    file_basename = \
                        os.path.basename(file_path).replace(f'.{extension}',
                                                            '')
                    if file_basename == align_basename:
                        return file_path

            return None

        for align_path in glob.glob(os.path.join(self.align_directory,
                                                 '*.align')):
            align_basename = os.path.basename(align_path).replace('.align', '')
            video_path = find_equivalent_file(self.video_directory, 'mpg')
            audio_path = find_equivalent_file(self.audio_directory, 'wav')

            yield align_path, video_path, audio_path

    def forced_alignment_prepare(self):
        """
        FA requires data in same directory unfortunately
        docker run -v /data:/shared --entrypoint /bin/bash -it liopa/forced-alignment
        https://github.com/prosodylab/Prosodylab-Aligner.git
        python3 -m aligner -r lang-mod.zip -a data/ -d lang.dict
        """

        for paths in tqdm(self.get_next_file_triple()):
            if any([path is None for path in paths]):
                continue

            align_path, video_path, audio_path = paths
            speaker_id = os.path.basename(os.path.dirname(video_path))

            new_lab_path = os.path.join(
                self.processed_data_directory,
                f"{speaker_id}_{os.path.basename(align_path).replace('align', 'lab')}"
            )
            new_audio_path = os.path.join(
                self.processed_data_directory,
                f'{speaker_id}_{os.path.basename(audio_path)}'
            )
            new_video_path = os.path.join(
                self.processed_data_directory,
                f'{speaker_id}_{os.path.basename(video_path)}'
            )

            if all([os.path.exists(new_path)
                    for new_path in [new_lab_path,
                                     new_audio_path,
                                     new_video_path]]):
                continue

            # get phrase from align path
            with open(align_path, 'r') as f:
                phrase = []
                for line in f.readlines():
                    line = line.strip()
                    word = str(line.split(' ')[-1]).upper()
                    if word not in ['SIL', 'SP']:
                        phrase.append(word)

            # write to lab path for FA
            with open(new_lab_path, 'w') as f:
                f.write(' '.join(phrase))

            # extract audio from video - can't use GRID audio
            command = [
                'ffmpeg',
                '-loglevel', 'error',
                '-i', f'{video_path}',
                '-ac', '1',  # ensure mono/1-channel audio
                f'{new_audio_path}'
            ]
            subprocess.call(command)

            # copy files to data directory - include speaker id in file names
            shutil.copy(video_path, new_video_path)


def main(args):
    dataset = GridDataset(**args.__dict__)
    getattr(dataset, args.run_type)()


if __name__ == '__main__':
    parser, sub_parsers = build_parser()

    main(parser.parse_args())
