import torch

# from ..registry import SAMPLERS
from .random_sampler import RandomSampler
from .sampling_result import SamplingResult


# @SAMPLERS.register
class PosNegBalanceSampler(RandomSampler):
    """Sampler pos and neg samples that num_pos = num_neg = min(num_pos, num_neg)"""

    def __init__(self, max_num, add_gt_as_proposals=True, **kwargs):
        self.max_num = max_num
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

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
        boxes = boxes[:, :4]

        gt_flags = boxes.new_zeros((boxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            boxes = torch.cat([gt_boxes, boxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = boxes.new_ones(gt_boxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_pos = (assign_result.gt_inds > 0).sum()
        num_neg = (assign_result.gt_inds == 0).sum()
        num_samples = min(num_pos, num_neg, self.max_num)
        num_samples = max(num_samples, 1)  # avoid empty sample results

        pos_inds = self.pos_sampler._sample_pos(assign_result, num_samples, boxes=boxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_samples, boxes=boxes, **kwargs)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, boxes, gt_boxes, assign_result, gt_flags)
