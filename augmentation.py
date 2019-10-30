import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms as tf
from IPython import embed

def resize(data, size):
    imgs, flows, inv_flows, masks, labels = data
    if size is not None:
        H, W = imgs.shape[-2:]
        imgs = F.interpolate(imgs, size=size, mode='bilinear')

        flows = F.interpolate(flows, size=size, mode='bilinear')
        flows[:, 0] = flows[:, 0] / W * size[1]
        flows[:, 1] = flows[:, 1] / H * size[0]

        inv_flows = F.interpolate(inv_flows, size=size, mode='bilinear')
        inv_flows[:, 0] = inv_flows[:, 0] / W * size[1]
        inv_flows[:, 1] = inv_flows[:, 1] / H * size[0]

        masks = F.interpolate(masks, size=size)
        labels = F.interpolate(labels, size=size)

    imgs = imgs.permute(1, 0, 2, 3)
    flows = flows.permute(1, 0, 2, 3)
    inv_flows = inv_flows.permute(1, 0, 2, 3)
    masks = masks.permute(1, 0, 2, 3)
    labels = labels.permute(1, 0, 2, 3)
    return imgs, flows, inv_flows, masks, labels

class Augmentation(object):
    def __init__(self, aug):
        super(Augmentation, self).__init__()
        self.aug = aug

    @staticmethod
    def _rotate(imgs, flows, inv_flows, masks, labels):
        B, _, H, W = imgs.size()
        p = 0.5
        param = 15
        if np.random.random() < p:
            angle = random.uniform(-param, param) * np.pi / 180
            theta = torch.zeros(B, 2, 3)
            theta[:, :, :2] = torch.tensor([[np.cos(angle), -1.0 * np.sin(angle)],
                                            [np.sin(angle), np.cos(angle)]])
            grid = F.affine_grid(theta, (B, 1, H, W))
            imgs = F.grid_sample(imgs, grid)

            flows = F.grid_sample(flows, grid)
            inv_flows = F.grid_sample(inv_flows, grid)
            # rotate flow value
            rot = theta[0, :, :2].t()
            flows = torch.matmul(flows.permute(0, 2, 3, 1), rot).permute(0, 3, 1, 2)
            inv_flows = torch.matmul(inv_flows.permute(0, 2, 3, 1), rot).permute(0, 3, 1, 2)

            masks = F.grid_sample(masks, grid, mode='nearest')
            labels = F.grid_sample(labels, grid, mode='nearest')
        return imgs, flows, inv_flows, masks, labels

    @staticmethod
    def _translate(imgs, flows, inv_flows, masks, labels):
        B, _, H, W = imgs.size()
        p = 0.5
        param = (0.05, 0.05)
        if np.random.random() < p:
            offset = (random.uniform(-param[0], param[0]) * 2, random.uniform(-param[1], param[1]) * 2)
            theta = torch.zeros(B, 2, 3)
            theta[:, :, :2] = torch.eye(2)
            theta[:, :, 2] = torch.tensor(offset)
            grid = F.affine_grid(theta, (B, 1, H, W))
            imgs = F.grid_sample(imgs, grid)
            flows = F.grid_sample(flows, grid)
            inv_flows = F.grid_sample(inv_flows, grid)
            masks = F.grid_sample(masks, grid, mode='nearest')
            labels = F.grid_sample(labels, grid, mode='nearest')
        return imgs, flows, inv_flows, masks, labels

    @staticmethod
    def _crop(imgs, flows, inv_flows, masks, labels):
        B, _, H, W = imgs.size()
        p = 0.5
        h, w = (360, 640)
        if H <= h or W <= w:
            return imgs, flows, inv_flows, masks, labels
        if np.random.random() < p:
            h0, w0 = random.randint(0, H - h), random.randint(0, W - w)
            imgs = imgs[..., h0:h0+h, w0:w0+w]
            flows = flows[..., h0:h0+h, w0:w0+w]
            inv_flows = inv_flows[..., h0:h0+h, w0:w0+w]
            masks = masks[..., h0:h0+h, w0:w0+w]
            labels = labels[..., h0:h0+h, w0:w0+w]
        return imgs, flows, inv_flows, masks, labels

    @staticmethod
    def _horizontal_flip(imgs, flows, inv_flows, masks, labels):
        p = 0.5
        if np.random.random() < p:
            imgs = torch.flip(imgs, dims=(-1,))

            flows = torch.flip(flows, dims=(-1,))
            inv_flows = torch.flip(inv_flows, dims=(-1,))
            flows[:, 0] = -flows[:, 0]
            inv_flows[:, 0] = -inv_flows[:, 0]

            masks = torch.flip(masks, dims=(-1,))
            labels = torch.flip(labels, dims=(-1,))
        return imgs, flows, inv_flows, masks, labels


    def color_warp(self, imgs, p=0.5):
        if not self.aug:
            return imgs
        if p < np.random.random():
            return imgs
        color = tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        transform = color.get_params(color.brightness, color.contrast,
                                     color.saturation, color.hue)
        return [transform(img) for img in imgs]


    def __call__(self, x):
        if not self.aug:
            return x
        x = self._rotate(*x)
        x = self._translate(*x)
        x = self._crop(*x)
        x = self._horizontal_flip(*x)
        return x
