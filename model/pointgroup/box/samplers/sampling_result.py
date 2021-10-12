import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, boxes, gt_boxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_boxes = boxes[pos_inds]
        self.neg_boxes = boxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_boxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_boxes = gt_boxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def boxes(self):
        return torch.cat([self.pos_boxes, self.neg_boxes])
