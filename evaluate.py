import torch

import os
from os.path import join
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from IPython import embed

from utils.IO import read, write

def evaluate(result_path):
    data_path = '/home/qzy/data/FBMS/Testset'
    data_dirs = sorted(glob(join(data_path, '*', 'GroundTruth')))
    result_dirs = sorted(glob(join(result_path, '*', 'label')))
    assert len(data_dirs) == len(result_dirs)
    avg_p, avg_r, avg_f, avg_delta_obj = [], [], [], []
    for data_dir, result_dir in zip(data_dirs, result_dirs):
        gt_img_list = sorted(glob(join(data_dir, '*.png')))
        gt_idx_list = list(map(lambda x: int(x.split('/')[-1][:3]) - 1, gt_img_list))
        if gt_idx_list[0] > 0:
            x = gt_idx_list[0]
            gt_idx_list = [y - x for y in gt_idx_list]
        label_list = sorted(glob(join(result_dir, '*.pfm')))

        gts, labels = [], []
        delta_obj = 0.
        for i, idx in enumerate(gt_idx_list[:-1]):
            gt = read(gt_img_list[i])
            if gt.ndim == 3:
                gt = gt[..., 0]
            label = read(label_list[idx]).astype(np.uint8)
            # embed()
            delta_obj += abs(len(np.unique(gt)) - len(np.unique(label)))
            gts.append(gt)
            labels.append(label)
        gts = np.stack(gts)
        labels = np.stack(labels)
        delta_obj /= (len(gt_idx_list) - 1)

        # objs = np.unique(gts)[1:]
        objs = np.unique(gts)
        n_obj = len(objs)
        regions = np.zeros(n_obj)
        for i, obj in enumerate(objs):
            regions[i] = np.sum(gts == obj)

        _labels = np.unique(labels)
        n_label = len(_labels)
        clusters = np.zeros(n_label)
        for j, label in enumerate(_labels):
            clusters[j] = np.sum(labels == label)

        # embed()
        overlap = np.zeros((n_obj, n_label))
        for i, obj in enumerate(objs):
            for j, label in enumerate(_labels):
                # embed()
                overlap[i, j] = np.sum((gts == obj) * (labels == label))

        P = overlap / clusters[None, :]
        R = overlap / regions[:, None]

        F = np.zeros_like(overlap)
        for i in range(n_obj):
            for j in range(n_label):
                if overlap[i, j] > 0.:
                    F[i, j] = (2 * P[i, j] * R[i, j]) / (P[i, j] + R[i, j])
        row_indices, col_indices = linear_sum_assignment(-F)

        precision, recall = 0., 0.
        for row, col in zip(row_indices, col_indices):
            precision += P[row, col]
            recall += R[row, col]
        for row in range(n_obj):
            if row not in row_indices:
                precision += 1.
        precision /= n_obj
        recall /= n_obj
        f_value = (2 * precision * recall) / (precision + recall)

        avg_p.append(precision)
        avg_r.append(recall)
        avg_f.append(f_value)
        avg_delta_obj.append(abs(n_obj - n_label))
        # avg_delta_obj.append(delta_obj)

    avg_p = np.mean(avg_p)
    avg_r = np.mean(avg_r)
    avg_f = np.mean(avg_f)
    avg_delta_obj = np.mean(avg_delta_obj)

    print(f"Model: {result_path.split('/')[-1]}")
    print("Precision: %f\tRecall: %f\tF-Score: %f\tDelta_Obj: %f" % (avg_p, avg_r, avg_f, avg_delta_obj))


if __name__ == '__main__':
    # evaluate on FBMS testset
    root = './result/experiment5'
    # result_path = './result/experiment4/epoch8/'
    result_paths = sorted(glob(join(root, '*')))
    for result_path in result_paths:
        evaluate(result_path)