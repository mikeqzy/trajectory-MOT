import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as tf
from glob import glob
import os
from os.path import join
import cv2
from PIL import Image
import numpy as np
from IPython import embed

from utils.IO import read
from flow.models import FlowNet2
from augmentation import Augmentation, resize

def generate_flow(model, img):
    # img: Tensor [3, T, H, W]
    img1 = img[None,:,:-1,...].permute(2, 1, 0, 3, 4)
    img2 = img[None,:,1:,...].permute(2, 1, 0, 3, 4)
    flow_input = torch.cat([img1, img2], dim=2).cuda()
    inv_flow_input = torch.cat([img2, img1], dim=2).cuda()

    with torch.no_grad():
        flow_output = model(flow_input).detach().cpu()
        inv_flow_output = model(inv_flow_input).detach().cpu()

    pad = torch.zeros(1, 2, *img.shape[-2:])
    flow_output = torch.cat([flow_output, pad], dim=0).permute(1, 0, 2, 3)
    inv_flow_output = torch.cat([pad, inv_flow_output], dim=0).permute(1, 0, 2, 3)
    return flow_output, inv_flow_output


class FT3D(Dataset):
    def __init__(self, root='/home/qzy/data/FT3D/', gt_flow=True, resize=(224, 400), T=5):
        super(FT3D, self).__init__()
        self.root = root
        # self.seq_start = self._get_seq_start()
        self.gt_flow = gt_flow
        self.resize = resize
        self.T = T

        path = join(self.root, 'trainList.txt')
        with open(path, 'r') as f:
            seq = sorted(set(map(lambda x: x[60:-10], f.readlines())))

        # seq = ['TRAIN/C/0616/right']

        self.dir_list = []
        for name in seq:
            image_dir = join(root, 'frames_cleanpass', name)
            flow_dir = join(root, 'optical_flow', *name.split('/')[:-1], 'into_future', name.split('/')[-1])
            inv_flow_dir = join(root, 'optical_flow', *name.split('/')[:-1], 'into_past', name.split('/')[-1])
            object_id_dir = join(root, 'object_index', name)
            mask_dir = join(root, 'motion_labels', name)
            self.dir_list.append((image_dir, flow_dir, inv_flow_dir, object_id_dir, mask_dir))


        # build flownet2.0
        if not self.gt_flow:
            class Dummy:
                def __init__(self):
                    self.rgb_max = 255.
                    self.fp16 = False

            args = Dummy()
            model = FlowNet2(args).cuda()
            resume = './flow/checkpoint/FlowNet2_checkpoint.pth'
            ckpt = torch.load(resume)
            model.load_state_dict(ckpt['state_dict'])
            self.model = model


    def __len__(self):
        return len(self.dir_list) * (10 - self.T + 1)

    def __getitem__(self, idx):
        T = self.T
        idx = idx // (10 - T + 1)
        start = idx % (10 - T + 1)

        idir, fdir, ifdir, oidir, mdir = self.dir_list[idx]
        img_names = sorted(glob(join(idir, '*.png')))[start:start + T]
        flow_names = sorted(glob(join(fdir, '*.pfm')))[start:start + T]
        inv_flow_names = sorted(glob(join(ifdir, '*.pfm')))[start:start + T]
        object_id_names = sorted(glob(join(oidir, '*.pfm')))[start:start + T]
        mask_names = sorted(glob(join(mdir, '*.png')))[start:start + T]

        try:
            # get image
            imgs = []
            for img_name in img_names:
                img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # transform = tf.Compose([
                #     tf.ToTensor(),
                #     tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                # ])
                transform = lambda x: torch.tensor(x).permute(2, 0, 1)
                img = transform(img).unsqueeze(0).float()
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            H, W = imgs.shape[-2:]
            if self.resize is not None:
                imgs = F.interpolate(imgs, size=self.resize, mode='bilinear').permute(1, 0, 2, 3)
            else:
                imgs = imgs.permute(1, 0, 2, 3)

            # get flow
            if self.gt_flow:
                flows = []
                for flow_name in flow_names:
                    flow = read(flow_name)
                    flow = torch.tensor(flow[...,:2].copy()).permute(2, 0, 1).unsqueeze(0)
                    flows.append(flow)
                # flows.append(torch.zeros_like(flows[0]))
                flows = torch.cat(flows, dim=0)

                inv_flows = []
                # inv_flows.append(torch.zeros_like(flows[:1]))
                for inv_flow_name in inv_flow_names:
                    inv_flow = read(inv_flow_name)
                    inv_flow = torch.tensor(inv_flow[...,:2].copy()).permute(2, 0, 1).unsqueeze(0)
                    inv_flows.append(inv_flow)
                inv_flows = torch.cat(inv_flows, dim=0)

                if self.resize is not None:
                    H, W = flows.shape[-2:]
                    flows = F.interpolate(flows, size=self.resize, mode='bilinear')
                    flows = flows.permute(1, 0, 2, 3)
                    flows[0] = flows[0] / W * self.resize[1]
                    flows[1] = flows[1] / H * self.resize[0]

                    inv_flows = F.interpolate(inv_flows, size=self.resize, mode='bilinear')
                    inv_flows = inv_flows.permute(1, 0, 2, 3)
                    inv_flows[0] = inv_flows[0] / W * self.resize[1]
                    inv_flows[1] = inv_flows[1] / H * self.resize[0]
                else:
                    flows = flows.permute(1, 0, 2, 3)
                    inv_flows = inv_flows.permute(1, 0, 2, 3)
            else:
                flows, inv_flows = generate_flow(self.model, imgs)

            # get mask
            masks = []
            for mask_name in mask_names:
                mask = cv2.imread(mask_name)[...,0]
                mask = torch.tensor(mask) > 0
                mask = mask.float()[None,None,...]
                masks.append(mask)
            masks = torch.cat(masks, dim=0)

            if self.resize is not None:
                masks = F.interpolate(masks, size=self.resize).permute(1, 0, 2, 3)
            else:
                masks = masks.permute(1, 0, 2, 3)

            # get label
            labels = []
            for label_name in object_id_names:
                label = read(label_name)
                label = torch.tensor(label.copy())[None,None,...]
                labels.append(label)
            labels = torch.cat(labels, dim=0)
            # print(labels.shape)

            if self.resize is not None:
                labels = F.interpolate(labels, size=self.resize).permute(1, 0, 2, 3)
            else:
                labels = labels.permute(1, 0, 2, 3)
            labels = masks * labels + (1 - masks) * -1
            objs = np.sort(np.unique(labels))
            # print(objs)
            n_clusters = len(objs) - 1
            for i, obj in enumerate(objs):
                labels[labels == obj] = i

            one_hot_labels = F.one_hot(labels.squeeze().long())
            one_hot_labels = one_hot_labels.permute(3, 0, 1, 2)[1:]
            assert not torch.isnan(imgs).any()
            assert not torch.isnan(flows).any()
            assert not torch.isnan(inv_flows).any()
            assert not torch.isnan(masks).any()
            assert not torch.isnan(one_hot_labels).any()

        except:
            with open('bad_data.txt', 'a+') as f:
                f.write(idir + '\n')
            return 0,0,0,0,0,0,0
        # print(one_hot_labels.shape)

        return imgs, flows, inv_flows, masks, one_hot_labels, n_clusters, idir

class FBMS(Dataset):
    def __init__(self, root='/home/qzy/data/FBMS/', resize=(224, 400), T=5, train=True, aug=True):
        super(FBMS, self).__init__()
        self.root = root
        self.resize = resize
        self.T = T
        self.aug = Augmentation(aug)

        if train:
            self.root = join(self.root, 'Trainingset')
        else:
            self.root = join(self.root, 'Testset')

        video_dirs = sorted(glob(join(self.root, '*')))
        self.dir_list = []
        self.video_len = []
        for video_dir in video_dirs:
            img_root = video_dir
            flow_root = join(video_dir, 'Flow')
            inv_flow_root = join(video_dir, 'Inv_Flow')
            label_root = join(video_dir, 'Label')
            n_img = len(sorted(glob(join(label_root, '*.pfm'))))
            self.video_len.append(n_img - T)
            self.dir_list.append([img_root, flow_root, inv_flow_root, label_root])

    def __len__(self):
        return np.sum(self.video_len)

    def __getitem__(self, idx):
        T = self.T
        i = 0
        while idx >= self.video_len[i]:
            idx -= self.video_len[i]
            i += 1
        img_dir, flow_dir, inv_flow_dir, label_dir = self.dir_list[i]
        img_names = sorted(glob(join(img_dir, '*.jpg')))[idx:idx + T]
        flow_names = sorted(glob(join(flow_dir, '*.flo')))[idx:idx + T]
        inv_flow_names = sorted(glob(join(inv_flow_dir, '*.flo')))[idx:idx + T]
        label_names = sorted(glob(join(label_dir, '*.pfm')))[idx:idx + T]

        try:
            # get image
            imgs = [Image.open(img_name) for img_name in img_names]
            imgs = self.aug.color_warp(imgs)
            transform = tf.Compose([
                tf.ToTensor(),
                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            imgs = [transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            # imgs = []
            # for img in _imgs:
            #     img = np.asarray(img)
            #     transform = lambda x: torch.tensor(x).permute(2, 0, 1)
            #     img = transform(img).unsqueeze(0).float()
            #     imgs.append(img)
            # imgs = torch.cat(imgs, dim=0)

            # get flow
            flows = []
            for flow_name in flow_names:
                flow = read(flow_name)
                flow = torch.tensor(flow).permute(2, 0, 1).unsqueeze(0)
                flows.append(flow)
            # flows.append(torch.zeros_like(flows[0]))
            flows = torch.cat(flows, dim=0)

            inv_flows = []
            # inv_flows.append(torch.zeros_like(flows[:1]))
            for inv_flow_name in inv_flow_names:
                inv_flow = read(inv_flow_name)
                inv_flow = torch.tensor(inv_flow).permute(2, 0, 1).unsqueeze(0)
                inv_flows.append(inv_flow)
            inv_flows = torch.cat(inv_flows, dim=0)

            # get label
            labels = []
            masks = []
            for label_name in label_names:
                label = read(label_name)
                label = torch.tensor(label.copy())[None, None, ...]
                mask = (label > 0).float()
                masks.append(mask)
                labels.append(label)
            masks = torch.cat(masks, dim=0)
            labels = torch.cat(labels, dim=0)
            objs = torch.unique(labels)
            # print(labels.shape)

            imgs, flows, inv_flows, masks, labels = resize(
                self.aug((imgs, flows, inv_flows, masks, labels)), self.resize)

            one_hot_labels = F.one_hot(labels.squeeze().long())
            one_hot_labels = one_hot_labels.permute(3, 0, 1, 2)[1:]
            n_clusters = one_hot_labels.size(0)

            assert not torch.isnan(imgs).any()
            assert not torch.isnan(flows).any()
            assert not torch.isnan(inv_flows).any()
            assert not torch.isnan(one_hot_labels).any()
            assert n_clusters > 0
        except:
            with open('bad_data.txt', 'a+') as f:
                f.write(img_dir + '\n')
            return 0,0,0,0,0,0,0

        return imgs, flows, inv_flows, masks, one_hot_labels, n_clusters, img_dir

class DAVIS_m(Dataset):
    def __init__(self, root='/home/qzy/data/DAVIS/', resize=(224, 400), T=5, aug=True):
        super(DAVIS_m, self).__init__()
        self.root = root
        self.resize = resize
        self.T = T

        self.aug = Augmentation(aug)

        path = join(root, 'ImageSets', 'DAVIS-m', 'train.txt')
        with open(path, 'r') as f:
            video_dirs = [name.strip() for name in f.readlines()]

        self.dir_list = []
        self.video_len = []
        for video_dir in video_dirs:
            img_root = join(root, 'JPEGImages', '480p', video_dir)
            flow_root = join(img_root, 'Flow')
            inv_flow_root = join(img_root, 'Inv_Flow')
            label_root = join(root, 'Annotations', '480p', video_dir)
            n_img = len(sorted(glob(join(img_root, '*.jpg'))))
            self.video_len.append(n_img - T)
            self.dir_list.append([img_root, flow_root, inv_flow_root, label_root])

    def __len__(self):
        return np.sum(self.video_len)

    def __getitem__(self, idx):
        T = self.T
        i = 0
        while idx >= self.video_len[i]:
            idx -= self.video_len[i]
            i += 1
        img_dir, flow_dir, inv_flow_dir, label_dir = self.dir_list[i]
        img_names = sorted(glob(join(img_dir, '*.jpg')))[idx:idx + T]
        flow_names = sorted(glob(join(flow_dir, '*.flo')))[idx:idx + T]
        inv_flow_names = sorted(glob(join(inv_flow_dir, '*.flo')))[idx:idx + T]
        label_names = sorted(glob(join(label_dir, '*.png')))[idx:idx + T]

        try:
            # get image
            imgs = [Image.open(img_name) for img_name in img_names]
            imgs = self.aug.color_warp(imgs)
            transform = tf.Compose([
                tf.ToTensor(),
                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            imgs = [transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            # imgs = []
            # for img in _imgs:
            #     img = np.asarray(img)
            #     transform = lambda x: torch.tensor(x).permute(2, 0, 1)
            #     img = transform(img).unsqueeze(0).float()
            #     imgs.append(img)
            # imgs = torch.cat(imgs, dim=0)

            # get flow
            flows = []
            for flow_name in flow_names:
                flow = read(flow_name)
                flow = torch.tensor(flow).permute(2, 0, 1).unsqueeze(0)
                flows.append(flow)
            # flows.append(torch.zeros_like(flows[0]))
            flows = torch.cat(flows, dim=0)

            inv_flows = []
            # inv_flows.append(torch.zeros_like(flows[:1]))
            for inv_flow_name in inv_flow_names:
                inv_flow = read(inv_flow_name)
                inv_flow = torch.tensor(inv_flow).permute(2, 0, 1).unsqueeze(0)
                inv_flows.append(inv_flow)
            inv_flows = torch.cat(inv_flows, dim=0)

            # get label
            labels = []
            for label_name in label_names:
                label = cv2.imread(label_name)
                label = torch.tensor(label).float()
                label = label[..., 0] + label[..., 1] * 255 + label[..., 2] * 255 * 255
                label = label[None, None, ...]
                labels.append(label)
            labels = torch.cat(labels, dim=0)
            objs = torch.unique(labels)
            for i, obj in enumerate(objs):
                labels[labels == obj] = i
            # print(labels.shape)

            masks = (labels > 0).float()

            imgs, flows, inv_flows, masks, labels = resize(
                self.aug((imgs, flows, inv_flows, masks, labels)), self.resize)

            one_hot_labels = F.one_hot(labels.squeeze().long())
            one_hot_labels = one_hot_labels.permute(3, 0, 1, 2)[1:]
            n_clusters = one_hot_labels.size(0)
            assert not torch.isnan(imgs).any()
            assert not torch.isnan(flows).any()
            assert not torch.isnan(inv_flows).any()
            assert not torch.isnan(one_hot_labels).any()
            assert n_clusters > 0
        except:
            with open('bad_data.txt', 'a+') as f:
                f.write(img_dir + '\n')
            return 0,0,0,0,0,0,0

        return imgs, flows, inv_flows, masks, one_hot_labels, n_clusters, img_dir

class ImagesFromFolder(Dataset):
    def __init__(self, args, root, iext='jpg'):
        self.args = args

        self.image_list = sorted(glob(os.path.join(root, '*.jpg')))
        self.frame_size = list(cv2.imread(self.image_list[0]).shape[:2])
        # padding to larger multiple
        self.render_size = list(map(lambda x: ((x - 1) // 64 + 1) * 64, self.frame_size))
        args.frame_size = self.frame_size
        args.render_size = self.render_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        idx = idx % len(self)
        img = cv2.imread(self.image_list[idx])
        if self.render_size != self.frame_size:
            pad_y = self.render_size[0] - self.frame_size[0]
            pad_x = self.render_size[1] - self.frame_size[1]
            img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, 0)
        np_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np_img.copy().transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))
        return img, np_img

if __name__ == "__main__":
    # dataset = torch.utils.data.ConcatDataset([DAVIS_m(), FBMS()])
    dataset = FBMS()
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        data = dataset[i]