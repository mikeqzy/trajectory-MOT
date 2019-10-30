import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tf

import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment
from IPython import embed

from dataset import *
from trajectory.dense_trajectory import DenseTrajectory
from mean_shift import vMF_MS
from loss import get_loss
from generate_flow import build_flow_model, generate_flow
from utils.IO import write

def inference_dataset(model_path='./ckpt/save/epoch3.pth'):
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sys.path.append(os.getcwd())

    print("Building Dataset...")
    dataset = FT3D()
    print("Finish building Dataset...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Building Model...")
    dt = DenseTrajectory().cuda()
    assert model_path is not None, "Pretrained model is not given"
    print("Loading Checkpoint...")
    ckpt = torch.load(model_path)
    model_dict = ckpt['model']
    dt.load_state_dict(model_dict)
    dt.eval()
    print("Finish Loading Checkpoint...")
    print("Finish Building Model")

    for i, data in enumerate(tqdm(dataloader)):
        imgs, flows, inv_flows, masks, labels, n_clusters, idir = data
        imgs, flows, inv_flows, masks, labels, n_clusters = \
            imgs.cuda(), flows.cuda(), inv_flows.cuda(), masks.cuda(), labels.cuda(), n_clusters.cuda()
        with torch.no_grad():
            fgmask, emb, tail = dt(imgs, flows, inv_flows)
            loss, _, _, dist_loss = get_loss(fgmask, emb, tail, masks, labels, n_clusters)
            print(loss, dist_loss)
            # embed()
            cluster = vMF_MS()
            sparse_label = cluster(emb, tail, n_clusters)
            label = dt.propagate(sparse_label)
            dt.visualize(label)

def load(img_names, flow_names, inv_flow_names, resize):
    imgs = [Image.open(img_name) for img_name in img_names]
    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    imgs = [transform(img) for img in imgs]
    imgs = torch.stack(imgs)
    size = imgs.shape[-2:]
    if resize is not None:
        imgs = F.interpolate(imgs, size=resize, mode='bilinear').permute(1, 0, 2, 3)
    else:
        imgs = imgs.permute(1, 0, 2, 3)
  
    flows = []
    for flow_name in flow_names:
        flow = read(flow_name)
        flow = torch.tensor(flow).permute(2, 0, 1)
        flows.append(flow)
    flows = torch.stack(flows)

    inv_flows = []
    for inv_flow_name in inv_flow_names:
        inv_flow = read(inv_flow_name)
        inv_flow = torch.tensor(inv_flow).permute(2, 0, 1)
        inv_flows.append(inv_flow)
    inv_flows = torch.stack(inv_flows)

    if resize is not None:
        H, W = flows.shape[-2:]
        flows = F.interpolate(flows, size=resize, mode='bilinear')
        flows = flows.permute(1, 0, 2, 3)
        flows[0] = flows[0] / W * resize[1]
        flows[1] = flows[1] / H * resize[0]

        inv_flows = F.interpolate(inv_flows, size=resize, mode='bilinear')
        inv_flows = inv_flows.permute(1, 0, 2, 3)
        inv_flows[0] = inv_flows[0] / W * resize[1]
        inv_flows[1] = inv_flows[1] / H * resize[0]
    else:
        flows = flows.permute(1, 0, 2, 3)
        inv_flows = inv_flows.permute(1, 0, 2, 3)

    return imgs, flows, inv_flows, size

def inference(video_dirs, window_size=5, iext='.jpg', resize=(224, 400), model_path='./ckpt/experiment4/epoch8.pth'):
    if isinstance(video_dirs, str):
        video_dirs = [video_dirs]
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sys.path.append(os.getcwd())

    print("Building Model...")
    dt = DenseTrajectory().cuda()
    assert model_path is not None, "Pretrained model is not given"
    print("Loading Checkpoint...")
    ckpt = torch.load(model_path)
    model_dict = ckpt['model']
    dt.load_state_dict(model_dict)
    dt.eval()
    print("Finish Loading Checkpoint...")
    print("Finish Building Model")

    for video_dir in tqdm(video_dirs):
        # check if flow precalculated
        flow_dir = join(video_dir, 'Flow')
        inv_flow_dir = join(video_dir, 'Inv_Flow')
        if not (os.path.exists(flow_dir) and os.path.exists(inv_flow_dir)):
            flow_model = build_flow_model()
            generate_flow(flow_model, video_dir)
        img_names = sorted(glob(join(video_dir, '*' + iext)))
        flow_names = sorted(glob(join(flow_dir, '*.flo')))
        inv_flow_names = sorted(glob(join(inv_flow_dir, '*.flo')))
        assert len(img_names) == len(flow_names) == len(inv_flow_names)

        save_path = join('./result', model_path.split('/')[-2],
                         model_path.split('/')[-1][:-4], video_dir.split('/')[-1])
        mask_save_path = join(save_path, 'mask')
        label_save_path = join(save_path, 'label')
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)

        # load and resize
        imgs, flows, inv_flows, size = load(img_names, flow_names, inv_flow_names, resize)

        # split by window size
        s_imgs = torch.split(imgs.unsqueeze(0), window_size, dim=2)
        s_flows = torch.split(flows.unsqueeze(0), window_size, dim=2)
        s_inv_flows = torch.split(inv_flows.unsqueeze(0), window_size, dim=2)

        trackers, tracker_id, max_id = None, None, None
        for clip_idx, (imgs, flows, inv_flows) in enumerate(zip(s_imgs, s_flows, s_inv_flows)):
            imgs, flows, inv_flows = imgs.cuda(), flows.cuda(), inv_flows.cuda()
            with torch.no_grad():
                fgmask, emb, tail = dt(imgs, flows, inv_flows)
                cluster = vMF_MS()
                try:
                    sparse_label, mean = cluster(emb, tail, torch.zeros(1))
                except:
                    embed()
                label = dt.propagate(sparse_label)

                # association between windows
                max_dist = 0.2
                if trackers is None:
                    trackers = mean.cpu().numpy()
                    N = trackers.shape[0]
                    tracker_id = np.arange(N)
                    max_id = N
                else:
                    objs = mean.cpu().numpy()
                    M, N = objs.shape[0], trackers.shape[0]
                    cost_matrix = np.zeros((M, N))
                    for i, obj in enumerate(objs):
                        for j, trk in enumerate(trackers):
                            dist = (1.0 - np.sum(obj * trk)) / 2.0
                            cost_matrix[i, j] = dist
                    # cost_matrix[cost_matrix >= max_dist] = max_dist

                    # print(cost_matrix)
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    # embed()

                    obj_id = [-1] * M
                    for row, col in zip(row_indices, col_indices):
                        if cost_matrix[row, col] < max_dist:
                            obj_id[row] = tracker_id[col]
                    for i in range(M):
                        if obj_id[i] < 0:
                            obj_id[i] = max_id
                            max_id += 1
                    trackers = objs
                    tracker_id = obj_id
                    # embed()

                masks, labels = dt.visualize(label, tracker_id)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(0)
                    labels = labels.unsqueeze(0)
                masks = F.interpolate(masks.permute(0, 3, 1, 2).float(), size=size).byte().permute(0, 2, 3, 1).numpy()
                labels = F.interpolate(labels.permute(0, 3, 1, 2).float(), size=size).permute(0, 2, 3, 1).numpy()
            for idx, (mask, label) in enumerate(zip(masks, labels)):
                index = clip_idx * window_size + idx
                cv2.imwrite(join(mask_save_path, f'{index:03d}.png'), mask)
                write(join(label_save_path, f'{index:03d}.pfm'), label)


if __name__ == '__main__':
    # inference_dataset()
    root = '/home/qzy/data/FBMS/Testset/'
    videos = sorted(glob(join(root, '*')))
    # videos = join(root, 'camel01')
    model_root = './ckpt/experiment5'
    models = [join(model_root, f'epoch{x}.pth') for x in range(5, 15)]
    for model in models:
        inference(videos, model_path=model)