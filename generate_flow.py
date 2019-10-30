import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
import cv2
import os
from os.path import join
from tqdm import tqdm
from IPython import embed

from flow.models import FlowNet2
from utils.IO import write, read

def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

def build_flow_model():
    class Dummy:
        def __init__(self):
            self.rgb_max = 255.
            self.fp16 = False

    args = Dummy()
    model = FlowNet2(args).cuda()
    resume = './flow/checkpoint/FlowNet2_checkpoint.pth'
    ckpt = torch.load(resume)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()
    return model

def generate_flow(model, video_dir, iext='.jpg'):
    frames = sorted(glob(join(video_dir, '*' + iext)))
    flow_path = join(video_dir, 'Flow')
    inv_flow_path = join(video_dir, 'Inv_Flow')
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)
    if not os.path.exists(inv_flow_path):
        os.makedirs(inv_flow_path)
    image_list = []
    for i in range(len(frames) - 1):
        image_list += [[frames[i], frames[i + 1]]]
    h, w = cv2.imread(frames[0]).shape[:2]
    h_, w_ = round2nearest_multiple(h, 64), round2nearest_multiple(w, 64)
    for i, pair in enumerate(tqdm(image_list)):
        _, name1 = os.path.split(pair[0])
        _, name2 = os.path.split(pair[1])
        img1 = cv2.imread(pair[0])[..., ::-1]
        img2 = cv2.imread(pair[1])[..., ::-1]
        images = [img1, img2]
        images = np.array(images).transpose((3, 0, 1, 2))
        images = torch.from_numpy(images.astype(np.float32))
        images = F.pad(images, [0, w_ - w, 0, h_ - h])
        inv_images = [img2, img1]
        inv_images = np.array(inv_images).transpose((3, 0, 1, 2))
        inv_images = torch.from_numpy(inv_images.astype(np.float32))
        inv_images = F.pad(inv_images, [0, w_ - w, 0, h_ - h])
        inp = torch.stack([images, inv_images])

        with torch.no_grad():
            output = model(inp.cuda()).detach().cpu()

        flow = output[0].numpy().transpose(1, 2, 0)
        inv_flow = output[1].numpy().transpose(1, 2, 0)
        flow, inv_flow = flow[:h, :w], inv_flow[:h, :w]
        write(join(flow_path, name1[:-4] + '.flo'), flow)
        write(join(inv_flow_path, name2[:-4] + '.flo'), inv_flow)
    padding = np.zeros((h, w, 2))
    _, name1 = os.path.split(frames[0])
    _, name2 = os.path.split(frames[-1])
    name1 = name1[:-4] + '.flo'
    name2 = name2[:-4] + '.flo'
    write(join(inv_flow_path, name1), padding)
    write(join(flow_path, name2), padding)

def main(dataset='FBMS'):
    model = build_flow_model()

    if dataset == 'FBMS':
        # FBMS
        root = '/home/qzy/data/FBMS/Trainingset'
        video_dirs = sorted(glob(join(root, '*')))
    elif dataset == 'DAVIS-m':
        path = '/home/qzy/data/DAVIS/ImageSets/DAVIS-m/train.txt'
        with open(path, 'r') as f:
            video_dirs = [name.strip() for name in f.readlines()]
        root = '/home/qzy/data/DAVIS/JPEGImages/480p/'
        video_dirs = [join(root, name) for name in video_dirs]
    else:
        raise RuntimeError
    for video_dir in tqdm(video_dirs):
        generate_flow(model, video_dir)


if __name__ == '__main__':
    main('DAVIS-m')