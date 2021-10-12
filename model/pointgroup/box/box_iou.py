import torch


def box_iou(boxes1, boxes2, aligned=True):
    """ Calculatte iou between two set of boxes

    Args:
        boxes1 (Tensor): shape (N, 6)
        boxes2 (Tensor): shape (M, 6)
        aligned (bool): if aligned: N = M

    Returns:
        iou (Tensor): shape (N, ) if aligned else (N, M)
    """
    if aligned:
        assert boxes1.size(0) == boxes2.size(0)
    else:
        boxes1 = boxes1[:, None, :]
        boxes2 = boxes2[None, :, :]

    start = torch.max(boxes1[..., :3], boxes2[..., :3])
    end = torch.min(boxes1[..., 3:], boxes2[..., 3:])

    intersection = (end - start).clamp(min=0).prod(-1)
    volume1 = (boxes1[..., 3:] - boxes1[..., :3]).prod(-1)
    volume2 = (boxes2[..., 3:] - boxes2[..., :3]).prod(-1)
    union = volume1 + volume2 - intersection

    iou = intersection / union
    return iou
