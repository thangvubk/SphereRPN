import numpy as np
import torch


def box2delta(proposals, gt, means=[0, 0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 3]) * 0.5
    py = (proposals[..., 1] + proposals[..., 4]) * 0.5
    pz = (proposals[..., 2] + proposals[..., 5]) * 0.5
    pw = proposals[..., 3] - proposals[..., 0]
    ph = proposals[..., 4] - proposals[..., 1]
    pd = proposals[..., 5] - proposals[..., 2]

    gx = (gt[..., 0] + gt[..., 3]) * 0.5
    gy = (gt[..., 1] + gt[..., 4]) * 0.5
    gz = (gt[..., 2] + gt[..., 5]) * 0.5
    gw = gt[..., 3] - gt[..., 0]
    gh = gt[..., 4] - gt[..., 1]
    gd = gt[..., 5] - gt[..., 2]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dz = (gz - pz) / pd
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dd = torch.log(gd / pd)
    deltas = torch.stack([dx, dy, dz, dw, dh, dd], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2box(rois,
              deltas,
              means=[0, 0, 0, 0, 0, 0],
              stds=[1, 1, 1, 1, 1, 1],
              max_shape=None,
              wh_ratio_clip=16 / 1000):
    """
    Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.

    Args:
        rois (Tensor): boxes to be transformed. Has shape (N, 6)
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
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 6)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 6)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::6]
    dy = denorm_deltas[:, 1::6]
    dz = denorm_deltas[:, 2::6]
    dw = denorm_deltas[:, 3::6]
    dh = denorm_deltas[:, 4::6]
    dd = denorm_deltas[:, 5::6]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    dd = dd.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 4]) * 0.5).unsqueeze(1).expand_as(dy)
    pz = ((rois[:, 2] + rois[:, 5]) * 0.5).unsqueeze(1).expand_as(dz)
    # Compute width/height of each roi
    pw = (rois[:, 3] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 4] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    pd = (rois[:, 5] - rois[:, 2]).unsqueeze(1).expand_as(dd)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gd = pd * dd.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gz = torch.addcmul(pz, 1, pd, dz)  # gz = pz + pd * dz
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    z1 = gz - gd * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    z2 = gz + gd * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[0])
        y1 = y1.clamp(min=0, max=max_shape[1])
        z1 = z1.clamp(min=0, max=max_shape[2])
        x2 = x2.clamp(min=0, max=max_shape[0])
        y2 = y2.clamp(min=0, max=max_shape[1])
        z2 = z2.clamp(min=0, max=max_shape[2])
    boxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1).view_as(deltas)
    return boxes


def get_centers_inside_boxes(centers, boxes):
    """
    Args:
        - centers (Tensor): shape (NA, NB, 3)
        - boxes (Tensor): shape (NA, NB, 6)
    Returns:
        - mask: shape (NA, NB), indicates whether centers is insdie boxes
    """
    cond1 = centers >= boxes[..., :3]
    cond2 = centers <= boxes[..., 3:]
    cond = cond1 & cond2
    mask = (cond[..., 0] & cond[..., 1] & cond[..., 2])
    return mask


def get_box_centers(boxes):
    """
    Args:
        - boxes (Tensor): shape (NA, 6)
    Returns:
        - centers (Tensor): shape (NA, 3)
    """
    return (boxes[:, :3] + boxes[:, 3:]) * 0.5


def get_distance(points1, points2):
    """Compute distance of 3d points"""
    d = ((points2 - points1)**2).sum(-1).sqrt()
    return d


def shift_boxes(boxes, sft):
    """
    Args:
        - boxes (Tensors): shape (..., 6)
        - sft (Tensor): shape (..., 3)
    Returns:
        - shifted_boxes (Tensor): shape (..., 6)
    """
    shifted_boxes = torch.zeros_like(boxes)
    shifted_boxes[..., :3] = boxes[..., :3] + sft
    shifted_boxes[..., 3:] = boxes[..., 3:] + sft
    return shifted_boxes


if __name__ == '__main__':
    # yapf: disable
    rois = torch.Tensor([[0., 0., 0., 1., 1., 1.],
                         [0., 0., 0., 1., 1., 1.],
                         [0., 0., 0., 1., 1., 1.],
                         [5., 6., 5., 6., 5., 6.]])

    gts = torch.Tensor([[1., 2., 3., 4., 5., 6.],
                        [0., 0., 0., 2., 2., 2.],
                        [1., 1., 1., 3., 4., 5.],
                        [5., 7., 5., 7., 5., 7.]])
    # yapdf: enable
    deltas = box2delta(rois, gts)
    boxes = delta2box(rois, deltas)
    assert (boxes == gts).sum() == gts.numel()
