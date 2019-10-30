import cv2
import numpy as np
import os
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from ynet.model import YNet, YNetPlus
from ptrnn.model import ConvRNN
from utils.misc import get_color

def cat(tensors, tensor):
    return torch.cat([tensors, tensor.unsqueeze(2)], dim=2)


class DenseTrajectory(nn.Module):
    def __init__(self, C=32):
        super(DenseTrajectory, self).__init__()
        self.ynet = YNetPlus(n_dim=C)
        self.ptrnn = ConvRNN(C=C)
        self.scm = nn.Sequential(
            nn.Linear(4, C, bias=False),
            # nn.GroupNorm(16, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(C, C, bias=False),
        )


    def forward(self, frames, flow, inv_flow, C=32):
        self.frames = frames  # [B, 3, T, H, W]
        self.flow = flow  # [B, 2, T, H, W]
        self.inv_flow = inv_flow  # [B, 2, T, H, W]
        self.C = C
        B, _, T, H, W = frames.shape
        self.B, self.C, self.T, self.H, self.W = B, C, T, H, W
        self.grid = self.get_meshgrid()

        # self.features = torch.zeros((B, C, T, H, W)).cuda()
        # self.fgmask = torch.zeros((B, 1, T, H, W)).cuda()
        #
        # self.vgrids = torch.zeros((B, 2, T, H, W)).cuda()
        # # consistency mask
        # self.mask = torch.zeros((B, 1, T, H, W)).cuda()
        # # end of trajectories
        # self.tails = torch.zeros((B, 1, T, H, W)).cuda()
        #
        # self.hidden_state = torch.zeros((B, C, T, H, W)).cuda()
        # self.cum_weight = torch.zeros((B, C, T, H, W)).cuda()
        self.track()
        emb = self.hidden_state / self.cum_weight.clamp(min=1e-12) + self.locs
        # emb = F.normalize(emb)
        return self.fgmask, emb, self.tails

    def get_meshgrid(self):
        # mesh grid
        B, H, W = self.B, self.H, self.W
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        return grid

    def get_vgrid(self, flow):
        vgrid = self.grid + flow

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(self.W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(self.H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        return vgrid

    def track(self):
        self.init()
        for idx in range(1, self.T):
            self.update(idx)
        self.get_trajectory()
        self.get_location()

    def init(self):
        feature, fgmask = self.ynet(self.frames[:,:,0,...], self.flow[:,:,0,...])
        self.features = feature.unsqueeze(2)
        self.fgmask = fgmask.unsqueeze(2)
        # self.vgrids = self.grid.unsqueeze(2)
        self.mask = (fgmask > 0.5).float().unsqueeze(2)

        pad = torch.zeros_like(feature).cuda()
        weight = self.ptrnn(pad, feature)
        self.hidden_state = (weight * feature).unsqueeze(2)
        self.cum_weight = weight.unsqueeze(2)

    def update(self, idx):
        feature, fgmask = self.ynet(self.frames[:,:,idx,...], self.flow[:,:,idx,...])
        self.features = cat(self.features, feature)
        self.fgmask = cat(self.fgmask, fgmask)
        fgmask = (fgmask > 0.5).float()
        vgrid = self.get_vgrid(self.inv_flow[:,:,idx,...])

        # get binary consistency mask
        # check if both are in foreground
        prev_fgmask = (self.fgmask[:,:,idx - 1,...] > 0.5).float()
        warped_fgmask = F.grid_sample(prev_fgmask, vgrid)
        warped_fgmask = (warped_fgmask > 0.5).float() # threshold due to bilinear interpolation
        mask = warped_fgmask * fgmask

        # check foreground/background consistency
        forward_flow = self.flow[:,:,idx - 1,...]
        warped_forward_flow = F.grid_sample(forward_flow, vgrid)
        backward_flow = self.inv_flow[:,:,idx,...]
        consistency_mask = torch.sum((warped_forward_flow + backward_flow) ** 2, dim=1, keepdim=True)  <= \
                           torch.sum(0.01 * (warped_forward_flow ** 2 + backward_flow ** 2), dim=1, keepdim=True) + 0.5
        mask = mask * consistency_mask.float()
        # TODO: check motion boundary?

        self.mask = cat(self.mask, mask)

        prev_hidden_state = self.hidden_state[:,:,idx - 1,...]
        warped_hidden_state = F.grid_sample(prev_hidden_state, vgrid) * mask
        prev_cum_weight = self.cum_weight[:,:,idx - 1,...]
        warped_cum_weight = F.grid_sample(prev_cum_weight, vgrid) * mask

        weight = self.ptrnn(warped_hidden_state / warped_cum_weight.clamp(min=1e-6), feature)

        self.hidden_state = cat(self.hidden_state, warped_hidden_state + weight * feature)
        self.cum_weight = cat(self.cum_weight, warped_cum_weight + weight)

    def get_trajectory(self):
        # get the tail of each trajectory
        tails = []
        for idx in range(self.T - 1):
            vgrid = self.get_vgrid(self.flow[:,:,idx,...])
            mask = self.mask[:,:,idx+1,...]
            mask = F.grid_sample(mask, vgrid)
            mask = (mask > 0.5).float()
            fgmask = (self.fgmask[:,:,idx,...] > 0.5).float()
            tail = ((fgmask - mask) == 1).float().unsqueeze(2)
            tails.append(tail)
        tails.append((self.fgmask[:,:,-1:,...] > 0.5).float())
        self.tails = torch.cat(tails, dim=2)

    def get_location(self):
        locs = []
        meshgrid = self.get_vgrid(torch.zeros_like(self.grid).cuda()).permute(0, 3, 1, 2)
        grid = torch.cat((meshgrid, self.flow[:,:,0],
                          torch.ones((self.B, 1, self.H, self.W)).cuda()), dim=1) # [B, 5, H, W]
        locs.append(grid)
        for idx in range(1, self.T):
            vgrid = self.get_vgrid(self.inv_flow[:,:,idx])
            mask = self.mask[:,:,idx]
            warped_grid = F.grid_sample(locs[-1], vgrid) * mask
            grid = torch.cat((meshgrid, self.flow[:,:,idx],
                              torch.ones((self.B, 1, self.H, self.W)).cuda()), dim=1)
            locs.append(warped_grid + grid)
        for loc in locs:
            loc[:, 0] /= loc[:, 4]
            loc[:, 1] /= loc[:, 4]
            loc[:, 2] /= loc[:, 4]
            loc[:, 3] /= loc[:, 4]
            # loc[:,2] = (loc[:,2] / loc[:,4] - loc[:,0] ** 2).clamp(min=0) ** 0.5
            # loc[:,3] = (loc[:,3] / loc[:,4] - loc[:,1] ** 2).clamp(min=0) ** 0.5
        locs = torch.stack(locs)[:,:,:4].permute((1, 0, 3, 4, 2)).reshape(-1, 4)
        locs = self.scm(locs)
        self.locs = locs.reshape(self.B, self.T, self.H, self.W, -1).permute((0, 4, 1, 2, 3))

    def propagate(self, sparse_label):
        # only for inference
        B, C, T, H, W = sparse_label.size()
        labels = [sparse_label[:, :, -1]]
        for idx in reversed(range(T - 1)):
            label = labels[-1]
            vgrid = self.get_vgrid(self.flow[:,:,idx])
            warped_label = F.grid_sample(label, vgrid)
            tail = self.tails[:,:,idx]
            fgmask = self.fgmask[:,:,idx] > 0.5
            label = warped_label * (1 - tail) + sparse_label[:,:,idx] * tail
            label = label * fgmask.float()
            labels.append(label)
        labels = torch.stack(labels[::-1], dim=2)
        labels = torch.argmax(labels, dim=1)
        return labels.squeeze().cpu()

    def visualize(self, label, tracker_id):
        # only for inference
        color = get_color()
        masks = torch.zeros(*label.size(), 3, dtype=torch.uint8)
        labels = torch.zeros_like(label)
        max_id = torch.max(label).item()
        for oid in range(max_id + 1):
            masks[label==oid] = torch.tensor(color[tracker_id[oid]])
            labels[label==oid] = torch.tensor([tracker_id[oid] + 1])
        masks = masks * (self.fgmask > 0.5).cpu().squeeze().unsqueeze(-1)
        labels = labels.float() * (self.fgmask > 0.5).cpu().squeeze().float()
        # embed()
        return masks, labels.unsqueeze(-1)
