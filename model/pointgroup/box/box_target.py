import torch
# from pcdet.utils.common_utils import multi_apply

from .transforms import box2delta


def box_target(pos_boxes_list,
               neg_boxes_list,
               pos_gt_boxes_list,
               pos_gt_labels_list,
               valid_roi_inds,
               target_means=[.0, .0, .0, .0, .0, .0],
               target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    labels, label_weights, box_targets, box_weights = multi_apply(
        box_target_single,
        pos_boxes_list,
        neg_boxes_list,
        pos_gt_boxes_list,
        pos_gt_labels_list,
        target_means=target_means,
        target_stds=target_stds)

    # concat target
    labels = torch.cat(labels, 0)
    label_weights = torch.cat(label_weights, 0)
    box_targets = torch.cat(box_targets, 0)
    box_weights = torch.cat(box_weights, 0)

    # ignore invalid target by setting weight to 0
    invalid_roi_mask = torch.ones_like(labels, dtype=torch.bool)
    valid_roi_inds = valid_roi_inds.long()
    invalid_roi_mask[valid_roi_inds] = 0
    label_weights[invalid_roi_mask] = 0.
    box_weights[invalid_roi_mask, :] = 0
    return labels, label_weights, box_targets, box_weights


def box_target_single(pos_boxes,
                      neg_boxes,
                      pos_gt_boxes,
                      pos_gt_labels,
                      target_means=[.0, .0, .0, .0, .0, .0],
                      target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_boxes.size(0)
    num_neg = neg_boxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_boxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_boxes.new_ones(num_samples)
    box_targets = pos_boxes.new_zeros(num_samples, 6)
    box_weights = pos_boxes.new_zeros(num_samples, 6)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_box_targets = box2delta(pos_boxes, pos_gt_boxes, target_means, target_stds)
        box_targets[:num_pos, :] = pos_box_targets
        box_weights[:num_pos, :] = 1.

    return labels, label_weights, box_targets, box_weights
