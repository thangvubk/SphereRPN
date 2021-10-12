import torch

# from ..registry import SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


# @SAMPLERS.register
class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, boxes, gt_boxes, **kwargs):
        boxes = boxes[:, :6]
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = boxes.new_zeros(boxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, boxes, gt_boxes, assign_result,
                                         gt_flags)
        return sampling_result
