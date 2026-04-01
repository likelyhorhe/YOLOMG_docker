import cv2
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--video_folder', type=str, default=os.path.join(PROJECT_ROOT, 'train_videos'), help='folder with input .mp4 files')
parser.add_argument('--image_folder', type=str, default=os.path.join(PROJECT_ROOT, 'train_images'), help='folder to save extracted frames')
opt = parser.parse_args()

video_folder = opt.video_folder
image_folder = opt.image_folder
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_image_folder = os.path.join(image_folder, video_file.split('.')[0])
    if not os.path.exists(video_image_folder):
        os.makedirs(video_image_folder)

    vc = cv2.VideoCapture(os.path.join(video_folder, video_file))
    c = 0
    rval = vc.isOpened()

    while rval:
        c += 1
        rval, frame = vc.read()
        if rval:
            name = str(c).zfill(4)
            cv2.imwrite(os.path.join(video_image_folder, video_file.split('.')[0] + '_' + name + '.jpg'), frame)
            print(f'extract frame from {video_file}:', name)
        else:
            break

    vc.release()
