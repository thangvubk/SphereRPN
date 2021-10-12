import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_ious, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_ious = max_ious
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_ious = torch.cat([self.max_ious.new_ones(self.num_gts), self.max_ious])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
