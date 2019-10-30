import torch
import torch.nn.functional as F
import numpy as np
from IPython import embed
# from scipy.special import iv



class vMF_MS(object):
    def __init__(self, kappa=10.):
        self.kappa = kappa

    def _random_sample(self, X, ratio=0.01, n_seeds=200, threshold=-0.3):
        # random sample far away points
        N, C = X.size()
        S = n_seeds
        if S >= N:
            return X
        samples = X[:1]
        X = X[1:]
        while samples.shape[0] < S:
            dot = self.matrix_dot(X, samples)
            # dot = dot.sum(1)
            dot = torch.max(dot, dim=1)[0]
            min_dot = torch.min(dot)
            samples = torch.cat([samples, X[dot == min_dot]], dim=0)
            X = X[~(dot == min_dot)]
        return samples

    @staticmethod
    def matrix_dot(x, y):
        return (x.unsqueeze(1) * y.unsqueeze(0)).sum(2)

    def _vMF_distribution(self, x):
        return torch.exp(self.kappa * x - self.kappa / 2)

    def _meanshift(self, Y, it=0, max_it=20):
        X = self.X
        N, C = X.size()
        S = Y.size(0)
        dot = self.matrix_dot(Y, X) # dot[i,j] = Y[i] dot X[j]
        weights = self._vMF_distribution(dot).unsqueeze(2).expand(S, N, C)
        X = X.unsqueeze(0).expand(S, N, C)
        new_Y = (weights * X).sum(1)
        new_Y = F.normalize(new_Y)
        if it >= max_it:
            return new_Y
        else:
            # print(((Y * new_Y).sum(1) < 1 - eps).sum())
            return self._meanshift(new_Y, it + 1)

    def _merge_centroids(self, centroids, min_dist=0.8):
        N, C = centroids.size()
        dot = self.matrix_dot(centroids, centroids) < min_dist
        # print('dot', dot.sum())
        mask = torch.ones(N).cuda().byte()
        centers = []
        for i in range(N):
            if mask[i]:
                centers.append(centroids[i])
                mask = mask * dot[i]
                # print(mask.sum())
                continue
        return torch.stack(centers)

    def __call__(self, emb: torch.Tensor, tail: torch.Tensor, n_clusters):
        B, C, T, H, W = emb.size()
        emb = F.normalize(emb)
        assert B == 1, "inference batch size should be 1"
        emb = emb.reshape(C, -1).permute(1, 0) # [V, C], V = T * H * W
        tail = tail.flatten().byte()
        X = emb[tail] # [N, C]
        self.X = X
        N = X.size(0)
        Y = self._random_sample(X)
        centroids = self._meanshift(Y)
        centroids = self._merge_centroids(centroids)
        K = centroids.size(0)
        # print(K, n_clusters.item())
        dot = self.matrix_dot(X, centroids)
        fg_label = torch.argmax(dot, dim=1)
        label = torch.zeros(T * H * W).cuda().long()
        label[tail] = fg_label + 1
        label = F.one_hot(label).permute(1, 0).reshape(B, -1, T, H, W)
        fg_label = F.one_hot(fg_label).unsqueeze(2).float()
        X = self.X.unsqueeze(1)
        X = X * fg_label
        mean = X.sum(0)
        mean = F.normalize(mean)
        return label[:,1:].float(), mean

    def test(self):
        def sample_vMF(mu, kappa, num_samples):
            """Generate num_samples N-dimensional samples from von Mises Fisher
            distribution around center mu \in R^N with concentration kappa.
            """
            dim = len(mu)
            result = np.zeros((num_samples, dim))
            for nn in range(num_samples):
                # sample offset from center (on sphere) with spread kappa
                w = _sample_weight(kappa, dim)

                # sample a point v on the unit sphere that's orthogonal to mu
                v = _sample_orthonormal_to(mu)

                # compute new point
                result[nn, :] = v * np.sqrt(1. - w ** 2) + w * mu

            return result

        def _sample_weight(kappa, dim):
            """Rejection sampling scheme for sampling distance from center on
            surface of the sphere.
            """
            dim = dim - 1  # since S^{n-1}
            b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
            x = (1. - b) / (1. + b)
            c = kappa * x + dim * np.log(1 - x ** 2)

            while True:
                z = np.random.beta(dim / 2., dim / 2.)
                w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
                u = np.random.uniform(low=0, high=1)
                if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                    return w

        def _sample_orthonormal_to(mu):
            """Sample point on sphere orthogonal to mu."""
            v = np.random.randn(mu.shape[0])
            proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
            orthto = v - proj_mu_v
            return orthto / np.linalg.norm(orthto)

        mu_0 = np.array([-0.251, -0.968, -0.105])
        mu_0 = mu_0 / np.linalg.norm(mu_0)
        mu_1 = np.array([0.399, 0.917, 0.713])
        mu_1 = mu_1 / np.linalg.norm(mu_1)
        mus = [mu_0, mu_1]
        kappa_0 = 10  # concentration parameter
        kappa_1 = 10  # concentration parameter
        kappas = [kappa_0, kappa_1]
        num_points_per_class = 300

        X_0 = sample_vMF(mu_0, kappa_0, num_points_per_class)
        X_1 = sample_vMF(mu_1, kappa_1, num_points_per_class)
        X = np.zeros((2 * num_points_per_class, 3))
        X[:num_points_per_class, :] = X_0
        X[num_points_per_class:, :] = X_1
        labels = np.zeros((2 * num_points_per_class,))
        labels[num_points_per_class:] = 1

        X = torch.tensor(X).cuda()
        centroids = self._meanshift(X, X)
        centroids = self._merge_centroids(centroids)
        embed()

class Gaussian_MS(object):
    def __init__(self, kappa=0.74):
        self.kappa = kappa

    def _random_sample(self, X, n_seeds=200):
        N, C = X.size()
        S = n_seeds
        if S >= N:
            return X
        samples = X[:1]
        X = X[1:]
        while samples.shape[0] < S:
            dist = self.matrix_dist(X, samples)
            dist = torch.min(dist, dim=1)
            min_dist = torch.min(dist)
            samples = torch.cat([samples, X[dist == min_dist]], dim=0)
            X = X[~(dist == min_dist)]
        return samples

    @staticmethod
    def matrix_dist(x, y):
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), p=2, dim=2)

    def _gaussian_distribution(self, x):
        return torch.exp(-x ** 2 / (2 * self.kappa ** 2))

    def _meanshift(self, Y, it=0, max_it=20):
        X = self.X
        N, C = X.size()
        S = Y.size(0)
        dist = self.matrix_dist(Y, X)
        weights = self._gaussian_distribution(dist).unsqueeze(2).expand(S, N, C)
        X = X.unsqueeze(0).expand(S, N, C)
        new_Y = (weights * X).mean(1)
        if it >= max_it:
            return new_Y
        else:
            return self._meanshift(new_Y, it + 1)

    def _merge_centroids(self, centroids, min_dist=1):
        N, C = centroids.size()
        dist = self.matrix_dist(centroids, centroids) < min_dist
        mask = torch.ones(N).cuda().byte()
        centers = []
        for i in range(N):
            if mask[i]:
                centers.append(centroids[i])
                mask = mask * dist[i]
                continue
        return torch.stack(centers)

    def __call__(self, emb: torch.Tensor, tail:torch.Tensor, n_clusters):
        B, C, T, H, W = emb.size()
        assert B == 1, "inference batch size should be 1"
        emb = emb.reshape(C, -1).permute(1, 0)  # [V, C], V = T * H * W
        tail = tail.flatten().byte()
        X = emb[tail]  # [N, C]
        self.X = X
        N = X.size(0)
        Y = self._random_sample(X)
        centroids = self._meanshift(Y)
        centroids = self._merge_centroids(centroids)
        K = centroids.size(0)
        # print(K, n_clusters.item())
        dist = self.matrix_dist(X, centroids)
        fg_label = torch.argmin(dist, dim=1)
        label = torch.zeros(T * H * W).cuda().long()
        label[tail] = fg_label + 1
        label = F.one_hot(label).permute(1, 0).reshape(B, -1, T, H, W)
        fg_label = F.one_hot(fg_label).unsqueeze(2).float()
        X = self.X.unsqueeze(1)
        X = X * fg_label
        mean = X.mean(0)
        return label[:, 1:].float(), mean

if __name__ == '__main__':
    cluster = vMF_MS()
    cluster.test()
