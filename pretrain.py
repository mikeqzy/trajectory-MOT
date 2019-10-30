from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import logging
from tqdm import tqdm
import numpy as np
from IPython import embed

from dataset import FT3D
from collator import Collator
from trajectory.dense_trajectory import DenseTrajectory
from loss import get_loss


def train(model_path=None):
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    sys.path.append(os.getcwd())

    print("Building Dataset...")
    dataset = FT3D()
    print("Finish building Dataset...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=Collator())
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    start_epoch = 1
    total_epoch = 10

    print("Building Model...")
    dt = DenseTrajectory().cuda()
    if model_path is not None:
        print("Loading Checkpoint...")
        ckpt = torch.load(model_path)
        model_dict = ckpt['model']
        dt.load_state_dict(model_dict, strict=False)
        start_epoch = ckpt['epoch'] + 1
        print("Finish Loading Checkpoint...")

    optimizer = torch.optim.SGD(dt.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    if torch.cuda.device_count() > 1:
        dt = nn.DataParallel(dt)
    print("Finish Building Model...")

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    show_interval = 20
    save_interval = 1000

    for epoch in range(start_epoch, total_epoch + 1):
        train_loss = []
        temp_loss, temp_fg_loss, temp_var_loss, temp_dist_loss = 0., 0., 0., 0.
        dt.train()
        for i, data in enumerate(tqdm(dataloader)):
            imgs, flows, inv_flows, masks, labels, n_clusters, _ = data
            if imgs.dim() == 1:
                continue
            imgs, flows, inv_flows, masks, labels, n_clusters = \
                imgs.cuda(), flows.cuda(), inv_flows.cuda(), masks.cuda(), labels.cuda(), n_clusters.cuda()
            # with torch.autograd.detect_anomaly():
            assert not torch.isnan(imgs).any()
            fgmask, emb, tail = dt(imgs, flows, inv_flows)
            assert not torch.isnan(emb).any()
            loss, fg_loss, var_loss, dist_loss = get_loss(fgmask, emb, tail, masks, labels, n_clusters)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dt.parameters(), 10.)
            optimizer.step()
            train_loss.append(loss.detach().cpu())
            temp_loss += loss.detach().cpu().item()
            # tqdm.write("temp_loss: %.4f loss: %.4f" % (temp_loss, loss.detach().cpu().item()))
            temp_fg_loss += fg_loss.detach().cpu().item()
            temp_var_loss += var_loss.detach().cpu().item()
            temp_dist_loss += dist_loss.detach().cpu().item()
            if (i + 1) % show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] loss: %.4f foreground_loss: %.4f intra_loss: %.4f inter_loss: %.4f"
                           % (epoch, i, temp_loss / show_interval, temp_fg_loss / show_interval,
                              temp_var_loss / show_interval, temp_dist_loss / show_interval))
                temp_loss, temp_fg_loss, temp_var_loss, temp_dist_loss = 0., 0., 0., 0.
            if (i + 1) % save_interval == 0:
                save_name = './ckpt/' + f'epoch{epoch}_iter{i + 1}.pth'
                new_state_dict = dt.state_dict()
                if torch.cuda.device_count() > 1:
                    new_state_dict = OrderedDict()
                    for k, v in dt.state_dict().items():
                        namekey = k[7:]  # remove `module.`
                        new_state_dict[namekey] = v

                torch.save({
                    'epoch': epoch,
                    'model': new_state_dict,
                    'optimizer': optimizer.state_dict(),
                }, save_name)
                print(f"Save Model: {save_name}")

        train_loss = np.mean(train_loss)
        print("EPOCH %d train_loss: %.4f" % (epoch, train_loss))
        save_name = './ckpt/' + f'epoch{epoch}.pth'
        new_state_dict = dt.state_dict()
        if torch.cuda.device_count() > 1:
            new_state_dict = OrderedDict()
            for k, v in dt.state_dict().items():
                namekey = k[7:]  # remove `module.`
                new_state_dict[namekey] = v

        torch.save({
            'epoch': epoch,
            'model': new_state_dict,
            'optimizer': optimizer.state_dict(),
        }, save_name)
        print(f"Save Model: {save_name}")

if __name__ == '__main__':
    train()