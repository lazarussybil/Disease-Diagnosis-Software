import os
import cv2
import random
import warnings
import argparse
import numpy as np
import seaborn as sns

from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_style('white')

def set_seed(seed = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def extractFrames(video_path: str, save_name: str, save_dir: str, n_frame_taken: int = 12) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_freq = int(cap.get(7) // n_frame_taken)
    i = 0
    success, data = cap.read()
    if not success:
        warnings.resetwarnings()
        warnings.warn(f'Video {video_path} unreadable!')
        warnings.filterwarnings('ignore')
        return
    while success:
        if i % frame_freq == 0:
            cv2.imwrite(os.path.join(save_dir, save_name + '_' + str(i // frame_freq) + '.jpg'), data)
        i += 1
        success, data = cap.read()
    cap.release()

def process_video(arg) -> None:
    for label in arg.labels:
        os.makedirs(os.path.join(arg.processed_video_dir, 'Train', label), exist_ok=True)
        os.makedirs(os.path.join(arg.processed_video_dir, 'Test', label), exist_ok=True)

    for label in arg.labels:
        file_dir = os.path.join(arg.raw_video_dir, label)
        num_total = len(os.listdir(file_dir))
        for ith, video_name in enumerate(tqdm(os.listdir(file_dir))):
            if ith < arg.train_ratio * num_total:
                extractFrames(
                    video_path=os.path.join(file_dir, video_name),
                    save_name=label + '_' + str(ith),
                    save_dir=os.path.join(arg.processed_video_dir, 'Train', label),
                    n_frame_taken=arg.n_frame_taken
                )
            else:
                extractFrames(
                    video_path=os.path.join(file_dir, video_name),
                    save_name=label + '_' + str(ith),
                    save_dir=os.path.join(arg.processed_video_dir, 'Test', label),
                    n_frame_taken=arg.n_frame_taken
                )

if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--labels', type = list, default = ['colitis', 'polyps'])
    parser.add_argument('--raw_video_dir', type = str, default = 'Data\\Raw\\Labeled_Videos_Co_Po')
    parser.add_argument('--processed_video_dir', type = str, default = 'Data\\Video_Images_Co_Po')
    parser.add_argument('--train_ratio', type = float, default = 0.8)
    parser.add_argument('--n_frame_taken', type = int, default = 64)
    process_video(parser.parse_args([]))


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--labels', type = list, default = ['BBPS-A', 'BBPS-B'])
    parser.add_argument('--raw_video_dir', type = str, default = 'Data\\Raw\\Labeled_Videos_BBPS')
    parser.add_argument('--processed_video_dir', type = str, default = 'Data\\Video_Images_BBPS')
    parser.add_argument('--train_ratio', type = float, default = 0.8)
    parser.add_argument('--n_frame_taken', type = int, default = 64)

    process_video(parser.parse_args([]))