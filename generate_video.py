import torch
import cv2
import numpy as np
from os.path import join
import os
from glob import glob
from tqdm import tqdm
from IPython import embed
from utils.IO import read
from utils.misc import get_color
from matplotlib import pyplot as plt

def visualize_label(label):
    color = get_color()
    mask = np.zeros(label.shape + (3, ))
    obj_id = np.unique(label)
    # embed()
    for obj in obj_id:
        if obj > 0:
            mask[label == obj] = color[int(obj)]
    return mask

def get_frames(video_dirs):
    if isinstance(video_dirs, str):
        video_dirs = [video_dirs]
    for video_dir in tqdm(video_dirs):
        img_dir = video_dir
        label_dir = join(video_dir, 'Label')
        result_dir = join('./result', 'epoch1', video_dir.split('/')[-1])
        save_path = join(result_dir, 'vis')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_list = sorted(glob(join(img_dir, '*.jpg')))
        label_list = sorted(glob(join(label_dir, '*.pfm')))
        result_list = sorted(glob(join(result_dir, '*.png')))
        # assert len(img_list) == len(label_list) == len(result_list)
        l = len(label_list)
        img_list, result_list = img_list[:l], result_list[:l]
        for i, (img, label, result) in enumerate(tqdm(zip(img_list, label_list, result_list), total=len(img_list))):
            img = cv2.imread(img)
            label = read(label)
            result = cv2.imread(result)
            masked_img = (result > 0) * result + (result == 0) * img
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(221)
            plt.imshow(img[:,:,::-1])
            bx = fig.add_subplot(222)
            plt.imshow(masked_img[:,:,::-1])
            cx = fig.add_subplot(223)
            plt.imshow(result[..., ::-1])
            dx = fig.add_subplot(224)
            plt.imshow(visualize_label(label).astype(np.uint8))
            plt.savefig(join(save_path, f'{i:03d}.jpg'))
            plt.close()

def generate_video(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    cmd = 'ffmpeg -r 12 -f image2 -s 800*800 -i %s -vcodec libx264 -crf 25 -pix_fmt yuv420p %s'
    for r_dir in dirs:
        i_dir = join(r_dir, '%03d.jpg')
        save_path = join(r_dir, '..', '..', 'videos')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        o_dir = join(save_path, r_dir.split('/')[-2] + '.mov')
        r_cmd = cmd % (i_dir, o_dir)
        # embed()
        os.system(r_cmd)



if __name__ == '__main__':
    root = '/home/qzy/data/FBMS/Trainingset'
    video_dirs = sorted(glob(join(root, '*')))
    get_frames(video_dirs)
    result_root = join('/home/qzy/codes/trajectory-MOT/result', 'epoch1')
    result_dirs = [join(result_root, os.path.split(x)[1], 'vis') for x in video_dirs]
    generate_video(result_dirs)