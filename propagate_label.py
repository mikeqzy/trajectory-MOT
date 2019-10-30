"""
Propagate the sparse GT label of FBMS dataset
"""
import torch
import torch.nn.functional as F
import cv2
from glob import glob
import os
from os.path import join
from tqdm import tqdm
from IPython import embed

from utils.IO import read, write

def main():
    root = '/home/qzy/data/FBMS/Trainingset'
    video_dirs = sorted(glob(join(root, '*')))

    for video_dir in tqdm(video_dirs):
        gt_root = join(video_dir, 'GroundTruth')
        save_root = join(video_dir, 'Label')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        gt_img_list = sorted(glob(join(gt_root, '*.png')))
        img_list = sorted(glob(join(video_dir, '*.jpg')))
        flow_list = sorted(glob(join(video_dir, 'Flow', '*.flo')))
        inv_flow_list = sorted(glob(join(video_dir, 'Inv_Flow', '*.flo')))
        gt_idx_list = list(map(lambda x: int(x.split('/')[-1][:3]) - 1, gt_img_list))
        if gt_idx_list[0] > 0:
            x = gt_idx_list[0]
            gt_idx_list = [y - x for y in gt_idx_list]
        n_clip = len(gt_idx_list) - 1
        gt = []
        for gt_img in gt_img_list:
            gt_img = cv2.imread(gt_img)[..., 0]
            gt_img = torch.tensor(gt_img).cuda()
            gt.append(gt_img)
        gt = torch.stack(gt)
        objs = torch.unique(gt, sorted=True)
        for i in range(n_clip):
            start = gt_idx_list[i]
            end = gt_idx_list[i + 1]
            start_gt = cv2.imread(gt_img_list[i])[..., 0]
            start_gt = torch.tensor(start_gt).cuda()
            end_gt = cv2.imread(gt_img_list[i + 1])[..., 0]
            end_gt = torch.tensor(end_gt).cuda()
            # objs = torch.unique(torch.stack([start_gt, end_gt]), sorted=True)
            for idx, obj in enumerate(objs):
                start_gt[start_gt == obj] = idx
                end_gt[end_gt == obj] = idx

            start_gt = F.one_hot(start_gt.long(), num_classes=objs.shape[0]).permute(2, 0, 1).unsqueeze(0).float()
            end_gt = F.one_hot(end_gt.long(), num_classes=objs.shape[0]).permute(2, 0, 1).unsqueeze(0).float()

            imgs, flows, inv_flows = [], [], []
            for t in range(start, end + 1):
                img = cv2.imread(img_list[t])
                # print(img_list[t])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).cuda()
                imgs.append(img)

                if t < end:
                    flow = read(flow_list[t])
                    # print(flow_list[t])
                    flow = torch.tensor(flow).permute(2, 0, 1).unsqueeze(0).cuda()
                    flows.append(flow)

                    inv_flow = read(inv_flow_list[t + 1])
                    # print(inv_flow_list[t])
                    inv_flow = torch.tensor(inv_flow).permute(2, 0, 1).unsqueeze(0).cuda()
                    inv_flows.append(inv_flow)

            H, W = start_gt.shape[-2:]
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W)
            yy = yy.view(1, 1, H, W)
            grid = torch.cat((xx, yy), 1).float().cuda()

            def get_vgrid(f):
                vgrid = grid + f

                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                return vgrid

            # forward propagate
            fwd_labels = [start_gt, ]
            for t in range(len(flows) - 1):
                fwd_flow = flows[t]
                bwd_flow = inv_flows[t]
                vgrid = get_vgrid(bwd_flow)
                warped_fwd_flow = F.grid_sample(fwd_flow, vgrid)
                mask = torch.sum((warped_fwd_flow + bwd_flow) ** 2, dim=1, keepdim=True) <= \
                                   torch.sum(0.01 * (warped_fwd_flow ** 2 + bwd_flow ** 2), dim=1, keepdim=True) + 0.5
                warped_label = F.grid_sample(fwd_labels[-1], vgrid) * mask.float()
                fwd_labels.append(warped_label)
            fwd_labels.append(torch.zeros_like(start_gt).cuda())

            # backward propagate
            bwd_labels = [end_gt, ]
            for t in reversed(range(1, len(flows))):
                fwd_flow = flows[t]
                bwd_flow = inv_flows[t]
                vgrid = get_vgrid(fwd_flow)
                warped_bwd_flow = F.grid_sample(bwd_flow, vgrid)
                mask = torch.sum((warped_bwd_flow + fwd_flow) ** 2, dim=1, keepdim=True) <= \
                       torch.sum(0.01 * (warped_bwd_flow ** 2 + fwd_flow ** 2), dim=1, keepdim=True) + 0.5
                warped_label = F.grid_sample(bwd_labels[-1], vgrid) * mask.float()
                bwd_labels.append(warped_label)
            bwd_labels.append(torch.zeros_like(end_gt).cuda())

            labels = []
            for fwd_label, bwd_label in zip(fwd_labels, reversed(bwd_labels)):
                label = fwd_label + bwd_label
                label = torch.argmax(label, dim=1).squeeze().float().cpu().numpy()
                labels.append(label)

            for idx, label in enumerate(labels):
                save_name = join(save_root, f'{start + idx + 1:03d}.pfm')
                write(save_name, label)

            print([x.sum() for x in labels])


if __name__ == '__main__':
    main()