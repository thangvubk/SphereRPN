import torch

from ..box_iou import box_iou
# from ..registry import ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


# @ASSIGNERS.register
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt box or background to each box.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive boxes.
        neg_iou_thr (float or tuple): IoU threshold for negative boxes.
        min_pos_iou (float): Minimum iou for a box to be considered as a
            positive box. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all boxes with the same
            highest iou with some gt to that gt.
    """

    def __init__(self, pos_iou_thr, neg_iou_thr, min_pos_iou=.0, gt_max_assign_all=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self, boxes, gt_boxes, gt_labels=None):
        """Assign gt to boxes.

        This method assign a gt box to every box (proposal/anchor), each box
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every box to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each box, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that box
        4. for each gt box, assign its nearest proposals (may be more than
           one) to itself

        Args:
            boxes (Tensor): Bounding boxes to be assigned, shape(n, 6).
            gt_boxes (Tensor): Groundtruth boxes, shape (k, 6).
            gt_labels (Tensor, optional): Label of gt_boxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            raise ValueError('No gt or boxes')
        boxes = boxes[:, :6]
        ious = box_iou(gt_boxes, boxes, False)

        assign_result = self.assign_wrt_ious(ious, gt_labels)
        return assign_result

    def assign_wrt_ious(self, ious, gt_labels=None):
        """Assign w.r.t. the ious of boxes with gts.

        Args:
            ious (Tensor): Overlaps between k gt_boxes and n boxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_boxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if ious.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_boxes = ious.size(0), ious.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = ious.new_full((num_boxes, ), -1, dtype=torch.long)

        # for each anchor, which gt best ious with it
        # for each anchor, the max iou of all gts
        max_ious, argmax_ious = ious.max(dim=0)
        # for each gt, which anchor best ious with it
        # for each gt, the max iou of all proposals
        gt_max_ious, gt_argmax_ious = ious.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_ious >= 0) & (max_ious < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_ious >= self.neg_iou_thr[0])
                             & (max_ious < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_ious >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_ious[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_ious[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = ious[i, :] == gt_max_ious[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_ious[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_boxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_ious, labels=assigned_labels)
