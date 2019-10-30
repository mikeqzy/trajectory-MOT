from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.02, delta_dist=0.5):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def forward(self, emb, tail, target, n_clusters):
        assert not target.requires_grad, "target should not have gradient"
        emb = F.normalize(emb)
        target = target.float()
        return self._discriminative_loss(emb, tail, target, n_clusters)

    def _discriminative_loss(self, emb, tail, target, n_clusters):
        B, C, _, _, _ = emb.size()
        N = target.size(1)
        emb = emb.reshape(B, C, -1)
        tail = tail.reshape(B, 1, -1)
        target = target.reshape(B, N, -1)
        c_means = self._cluster_means(emb, tail, target, n_clusters)
        l_var = self._variance_term(emb, tail, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)

        return l_var, l_dist

    @staticmethod
    def _cluster_means(emb, tail, target, n_clusters):
        B, C, V = emb.size()
        N = target.size(1)

        emb = emb.unsqueeze(2).expand(B, C, N, V)
        target = target.unsqueeze(1)
        tail = tail.unsqueeze(2)
        emb = emb * target * tail

        means = []
        for i in range(B):
            emb_sample = emb[i, :, :n_clusters[i]]
            # target_sample = target[i, :, :n_clusters[i]]
            mean_sample = emb_sample.sum(2)
            # mean_sample = mean_sample / mean_sample.norm(p=2, dim=0, keepdim=True).clamp(min=1e-10)
            mean_sample = F.normalize(mean_sample, dim=0)

            n_pad_c = N - n_clusters[i]
            assert n_pad_c >= 0
            if n_pad_c > 0:
                pad_sample = torch.zeros(C, n_pad_c).cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        means = torch.stack(means)
        return means


    def _variance_term(self, emb, tail, target, c_means, n_clusters):
        B, C, V = emb.size()
        N = target.size(1)
        try:
            c_means = c_means.unsqueeze(3).expand(B, C, N, V)
            emb = emb.unsqueeze(2).expand(B, C, N, V)
            tail = tail.expand(B, N, V)
            var = ((torch.ones(B, N, V).cuda() - (c_means * emb).sum(1)) / 2.0) * target * tail

            var_term = torch.zeros(1).cuda()
            for i in range(B):
                var_sample = var[i, :n_clusters[i]]
                # target_sample = target[i, :n_clusters[i]]
                mask_sample = var_sample.ge(self.delta_var).float()
                var_sample = mask_sample * var_sample
                var_sample = var_sample ** 2
                c_var = var_sample.sum(1) / mask_sample.sum(1).clamp(min=50)
                var_term += c_var.sum() / n_clusters[i]
            var_term /= B
            assert not torch.isnan(var_term)
        except:
            embed()

        return var_term


    def _distance_term(self, c_means, n_clusters):
        bs, n_f, max_n_c = c_means.size()

        dist_term = torch.zeros(1).cuda()
        for i in range(bs):
            n_c = n_clusters[i]
            if n_c <= 1:
                continue

            # n_f, n_c
            mean_sample = c_means[i, :, :n_c]

            means_a = mean_sample.unsqueeze(2).expand(n_f, n_c, n_c)
            means_b = means_a.permute(0, 2, 1)
            dist = (torch.ones(n_c, n_c).cuda() - (means_a * means_b).sum(0)) / 2.0
            margin = -self.delta_dist * (torch.eye(n_c).float().cuda() - 1.0)
            c_dist = ((margin - dist).clamp(min=0) ** 2).sum()
            dist_term += c_dist / (n_c * (n_c - 1))
        dist_term /= bs

        return dist_term

class DiscriminativeLossL2(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=1.5):
        super(DiscriminativeLossL2, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def forward(self, emb, tail, target, n_clusters):
        assert not target.requires_grad, "target should not have gradient"
        target = target.float()
        return self._discriminative_loss(emb, tail, target, n_clusters)

    def _discriminative_loss(self, emb, tail, target, n_clusters):
        B, C, _, _, _ = emb.size()
        N = target.size(1)
        emb = emb.reshape(B, C, -1)
        tail = tail.reshape(B, 1, -1)
        target = target.reshape(B, N, -1)
        c_means = self._cluster_means(emb, tail, target, n_clusters)
        l_var = self._variance_term(emb, tail, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)

        return l_var, l_dist, l_reg

    @staticmethod
    def _cluster_means(emb, tail, target, n_clusters):
        B, C, V = emb.size()
        N = target.size(1)

        emb = emb.unsqueeze(2).expand(B, C, N, V)
        target = target.unsqueeze(1)
        tail = tail.unsqueeze(2)
        emb = emb * target * tail

        means = []
        for i in range(B):
            emb_sample = emb[i, :, :n_clusters[i]]
            # target_sample = target[i, :, :n_clusters[i]]
            mean_sample = emb_sample.mean(2)

            n_pad_c = N - n_clusters[i]
            assert n_pad_c >= 0
            if n_pad_c > 0:
                pad_sample = torch.zeros(C, n_pad_c).cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        means = torch.stack(means)
        return means


    def _variance_term(self, emb, tail, target, c_means, n_clusters):
        B, C, V = emb.size()
        N = target.size(1)
        # try:
        c_means = c_means.unsqueeze(3).expand(B, C, N, V)
        emb = emb.unsqueeze(2).expand(B, C, N, V)
        tail = tail.expand(B, N, V)
        var = ((c_means - emb) ** 2).sum(1) * target * tail

        var_term = torch.zeros(1).cuda()
        for i in range(B):
            var_sample = var[i, :n_clusters[i]]
            # target_sample = target[i, :n_clusters[i]]
            mask_sample = var_sample.ge(self.delta_var ** 2).float()
            var_sample = mask_sample * var_sample
            c_var = var_sample.sum(1) / mask_sample.sum(1).clamp(min=50)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= B
        assert not torch.isnan(var_term)
        # except:
        #     embed()

        return var_term


    def _distance_term(self, c_means, n_clusters):
        bs, n_f, max_n_c = c_means.size()

        dist_term = torch.zeros(1).cuda()
        for i in range(bs):
            n_c = n_clusters[i]
            if n_c <= 1:
                continue

            # n_f, n_c
            mean_sample = c_means[i, :, :n_c]

            means_a = mean_sample.unsqueeze(2).expand(n_f, n_c, n_c)
            means_b = means_a.permute(0, 2, 1)
            dist = (torch.ones(n_c, n_c).cuda() - (means_a * means_b).sum(0)) / 2.0
            dist = ((means_a - means_b) ** 2).sum(0) ** 0.5
            margin = -self.delta_dist * (torch.eye(n_c).float().cuda() - 1.0)
            c_dist = ((margin - dist).clamp(min=0) ** 2).sum()
            dist_term += c_dist / (n_c * (n_c - 1))
        dist_term /= bs

        return dist_term

    @staticmethod
    def _regularization_term(c_means, n_clusters):
        B, C, N = c_means.size()

        reg_term = torch.zeros(1).cuda()
        for i in range(B):
            n_c = n_clusters[i]
            mean_sample = c_means[i, :, :n_c]
            reg_term += torch.mean(torch.norm(mean_sample, 2, 0))
        reg_term /= B

        return reg_term


def get_loss(fgmask, emb, tail, gt_mask, gt_label, n_clusters, alpha=1, beta=1, gamma=1):
    fg_crit = nn.BCELoss()
    dis_crit = DiscriminativeLoss()
    # foreground loss
    fg_loss = fg_crit(fgmask, gt_mask) # [B, 1, T, H, W]
    # discriminative loss
    var_loss, dist_loss = dis_crit(emb, tail, gt_label, n_clusters) # [B, max_n_clusters, T, H, W]
    return alpha * fg_loss + beta * var_loss + gamma * dist_loss, fg_loss, var_loss, dist_loss
    # return beta * var_loss + gamma * dist_loss, fg_loss, var_loss, dist_loss

def get_l2_loss(fgmask, emb, tail, gt_mask, gt_label, n_clusters, alpha=1, beta=1, gamma=1, phi=0.001):
    fg_crit = nn.BCELoss()
    dis_crit = DiscriminativeLossL2()
    # foreground loss
    fg_loss = fg_crit(fgmask, gt_mask)  # [B, 1, T, H, W]
    # discriminative loss
    var_loss, dist_loss, reg_loss = dis_crit(emb, tail, gt_label, n_clusters)  # [B, max_n_clusters, T, H, W]
    return alpha * fg_loss + beta * var_loss + gamma * dist_loss + phi * reg_loss, fg_loss, var_loss, dist_loss