import torch
from IPython import embed

class Collator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if not isinstance(x[0], int)]
        if len(batch) == 0:
            return torch.zeros(1), 0, 0, 0, 0, 0, 0
        transposed_batch = list(zip(*batch))
        imgs, flows, inv_flows, masks, labels, n_clusters, idirs = transposed_batch
        imgs = torch.stack(imgs)
        flows = torch.stack(flows)
        inv_flows = torch.stack(inv_flows)
        masks = torch.stack(masks)
        labels = self._pad_cluster(labels)
        n_clusters = torch.tensor(n_clusters)
        return imgs, flows, inv_flows, masks, labels, n_clusters, idirs

    @staticmethod
    def _pad_cluster(tensors):
        max_n_clusters = max([tensor.shape[0] for tensor in tensors])
        batch_shape = (len(tensors), max_n_clusters, ) + tuple(tensors[0].shape[-3:])
        batched_labels = tensors[0].new(*batch_shape).zero_()
        for label, pad_label in zip(tensors, batched_labels):
            pad_label[:label.shape[0]].copy_(label)
        return batched_labels
