import numpy as np
import torch
from .utils import multi_apply

# from .box import box2delta, build_assigner, build_sampler
from .box.assigners import CenterAssigner
from .box.samplers import PosNegBalanceSampler

INF = 1e10


def sphere2delta(spheres, gts, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert spheres.size() == gts.size()

    spheres = spheres.float()
    gts = gts.float()
    # px = (proposals[..., 0] + proposals[..., 3]) * 0.5
    # py = (proposals[..., 1] + proposals[..., 4]) * 0.5
    # pz = (proposals[..., 2] + proposals[..., 5]) * 0.5
    # pw = proposals[..., 3] - proposals[..., 0]
    # ph = proposals[..., 4] - proposals[..., 1]
    # pd = proposals[..., 5] - proposals[..., 2]

    # gx = (gt[..., 0] + gt[..., 3]) * 0.5
    # gy = (gt[..., 1] + gt[..., 4]) * 0.5
    # gz = (gt[..., 2] + gt[..., 5]) * 0.5
    # gw = gt[..., 3] - gt[..., 0]
    # gh = gt[..., 4] - gt[..., 1]
    # gd = gt[..., 5] - gt[..., 2]

    # dx = (gx - px) / pw
    # dy = (gy - py) / ph
    # dz = (gz - pz) / pd
    # dw = torch.log(gw / pw)
    # dh = torch.log(gh / ph)
    # dd = torch.log(gd / pd)
    d_centers = (gts[..., :3] - spheres[..., :3]) / spheres[..., 3:4]
    d_radius = torch.log(gts[..., 3:4] / spheres[..., 3:4])
    deltas = torch.cat([d_centers, d_radius], dim=-1)
    # deltas = torch.stack([dx, dy, dz, dw, dh, dd], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2sphere(rois,
                 deltas,
                 means=[0, 0, 0, 0],
                 stds=[1, 1, 1, 1],
                 max_shape=None,
                 wh_ratio_clip=16 / 1000):
    """
    Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.

    Args:
        rois (Tensor): boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): encoded offsets with respect to each roi.
            Has shape (N, 6). Note N = num_anchors * W * H when rois is a grid
            of anchors. Offset encoding follows [1]_.
        means (list): denormalizing means for delta coordinates
        stds (list): denormalizing standard deviation for delta coordinates
        max_shape (tuple[int, int]): maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): maximum aspect ratio for boxes.

    Returns:
        Tensor: boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dz = denorm_deltas[:, 2::4]
    dr = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dr = dr.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = rois[:, 0:1].expand_as(dx)
    py = rois[:, 1:2].expand_as(dy)
    pz = rois[:, 2:3].expand_as(dz)
    pr = rois[:, 3:4].expand_as(dr)
    # Use exp(network energy) to enlarge/shrink each roi
    gr = pr * dr.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pr, dx)  # gx = px + pr * dx
    gy = torch.addcmul(py, 1, pr, dy)  # gy = py + ph * dy
    gz = torch.addcmul(pz, 1, pr, dz)  # gz = pz + pd * dz
    # Convert center-xy/width/height to top-left, bottom-right
    out_spheres = torch.stack([gx, gy, gz, gr], dim=-1).view_as(deltas)
    return out_spheres

def anchor_target(anchor_list, gt_box_list, volume_ranges, cfg):
    """Compute classification, box regression targets for anchors
    Args:
        - anchor_list list[list[Tensor]]: per level, per scan anchors
        - gt_box_list list(Tensor): ground truth box of each scan
        - volume_range list(tuple): volume range of gt box each level
    Returns:
        - flatten_anchors (Tensor)
        - flatten_label_targets (Tensor)
        - flatten_box_targets (Tensor)
    """
    # transpose per-level-per-scan to per-scan-per-level
    tp_anchor_list = list(zip(*anchor_list))

    # process anchors of each scan
    num_scans = len(tp_anchor_list)
    num_anchor_list = []
    volume_range_list = []
    for i in range(num_scans):
        num_anchor_list.append([anchor.size(0) for anchor in tp_anchor_list[i]])
        volume_range_per_scan = [
            anchor.new_tensor(regression_range)[None].expand(anchor.size(0), 2)
            for anchor, regression_range in zip(tp_anchor_list[i], volume_ranges)
        ]
        volume_range_per_scan = torch.cat(volume_range_per_scan, dim=0)
        volume_range_list.append(volume_range_per_scan)
        tp_anchor_list[i] = torch.cat(tp_anchor_list[i], dim=0)

    target_results = multi_apply(
        anchor_target_single, tp_anchor_list, gt_box_list, volume_range_list, cfg=cfg)
    (label_target_list, label_weight_list, box_target_list, box_weight_list, num_pos_list,
     num_neg_list) = target_results

    total_num_pos = max(sum(num_pos_list), 1)
    total_num_neg = max(sum(num_neg_list), 1)

    # split to per scan, per level
    label_target_list = split(label_target_list, num_anchor_list)
    label_weight_list = split(label_weight_list, num_anchor_list)
    box_target_list = split(box_target_list, num_anchor_list)
    box_weight_list = split(box_weight_list, num_anchor_list)

    # transpose back to per level, per scan
    label_target_list = list(zip(*label_target_list))
    label_weight_list = list(zip(*label_weight_list))
    box_target_list = list(zip(*box_target_list))
    box_weight_list = list(zip(*box_weight_list))

    # concat to single tensor
    flatten_anchors = concat_list_of_list(anchor_list)
    flatten_label_targets = concat_list_of_list(label_target_list)
    flatten_label_weights = concat_list_of_list(label_weight_list)
    flatten_box_targets = concat_list_of_list(box_target_list)
    flatten_box_weights = concat_list_of_list(box_weight_list)
    return (flatten_anchors, flatten_label_targets, flatten_label_weights, flatten_box_targets,
            flatten_box_weights, total_num_pos, total_num_neg)


def anchor_target_single(anchors, gt_boxes, volume_ranges, cfg):
    """
    Args:
        - anchors (Tensor): shape (NA, 6)
        - gt_boxes (Tensor): shape (NB, 6)
    Returns:
        - label_targets (Tensor): shape (NA, 1)
        - box_targets (Tensor): shape (NA, 6)
    """
    cfg = dict(
        assigner=dict(type='CenterAssigner'),
        # sampler=dict(type='PseudoSampler')),
        sampler=dict(type='PosNegBalanceSampler', add_gt_as_proposals=False))
    num_anchors = anchors.size(0)
    # assigner = build_assigner(cfg.assigner)
    # sampler = build_sampler(cfg.sampler)
    assigner = CenterAssigner()
    sampler = PosNegBalanceSampler(max_num=128, add_gt_as_proposals=False)
    assign_result = assigner.assign(anchors, gt_boxes, volume_ranges)
    sampling_result = sampler.sample(assign_result, anchors, gt_boxes)

    box_targets = torch.zeros_like(anchors)
    box_weights = torch.zeros_like(anchors)
    label_targets = anchors.new_zeros(num_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_anchors)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_box_targets = sphere2delta(sampling_result.pos_boxes, sampling_result.pos_gt_boxes)
        box_targets[pos_inds, :] = pos_box_targets
        box_weights[pos_inds, :] = 1.0
        label_targets[pos_inds] = 1
        label_weights[pos_inds] = 1.0
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    num_pos = pos_inds.size(0)
    num_neg = sampling_result.neg_inds.size(0)

    return label_targets, label_weights, box_targets, box_weights, num_pos, num_neg


def concat_list_of_list(tensor_list):
    """Concat list[list[Tensor]] to a single tensor"""
    tensor_list = [torch.cat(tensors, dim=0) for tensors in tensor_list]
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


def split(target_list, num_anchor_list):
    """For each target and num_anchors: split target to a list of sub-target"""
    target_list = [
        label_targets.split(num_anchors, 0)
        for label_targets, num_anchors in zip(target_list, num_anchor_list)
    ]
    return target_list
