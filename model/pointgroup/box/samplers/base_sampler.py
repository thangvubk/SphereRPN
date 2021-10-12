from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self, assign_result, boxes, gt_boxes, gt_labels=None, **kwargs):
        """Sample positive and negative boxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth boxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            boxes (Tensor): Boxes to be sampled from.
            gt_boxes (Tensor): Ground truth boxes.
            gt_labels (Tensor, optional): Class labels of ground truth boxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        boxes = boxes[:, :6]

        gt_flags = boxes.new_zeros((boxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            boxes = torch.cat([gt_boxes, boxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = boxes.new_ones(gt_boxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, boxes=boxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, boxes=boxes, **kwargs)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, boxes, gt_boxes, assign_result, gt_flags)
