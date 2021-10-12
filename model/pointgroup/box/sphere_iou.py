import torch
import numpy as np


def sphere_iou(spheres1, spheres2, aligned=True, eps=1e-6):
    """ Calculatte iou between two set of spheres

    Args:
        spheres1 (Tensor): shape (N, 4)
        spheres2 (Tensor): shape (M, 4)
        aligned (bool): if aligned: N = M

    Returns:
        iou (Tensor): shape (N, ) if aligned else (N, M)
    """
    assert isinstance(spheres1, torch.Tensor)
    assert isinstance(spheres2, torch.Tensor)
    N = spheres1.size(0)
    M = spheres2.size(0)
    assert N > 0 and M > 0

    if aligned:
        assert N == M
    else:
        spheres1 = spheres1[:, None, :].expand(N, M, 4)
        spheres2 = spheres2[None, :, :].expand(N, M, 4)

    center1 = spheres1[..., :3]
    center2 = spheres2[..., :3]
    r1 = spheres1[..., 3]
    r2 = spheres2[..., 3]
    sum_volume = (4*r1**3 + 4*r2**3) / 3

    if isinstance(center1, np.ndarray):
        d = np.linalg.norm(center1 - center2, ord=2, axis=-1)
    else:
        d = torch.norm(center1 - center2, p=2, dim=-1)

    # dispatch IoU computation base one r1, r2 and d
    # default computation
    intersection = (r1 + r2 - d)**2 * (d**2 + 2*d*(r1 + r2) - 3 * (r1 - r2)**2) / (12 * d)
    # d >= r1 + r2: two spheres have no intersection
    inds = (d >= (r1 + r2))
    if inds.any():
        intersection[inds] = 0
    # d <= |r1 - r2|: one sphere inside the other:
    inds = (d <= (r1 - r2))
    if inds.any():
        intersection[inds] = 4*r2[inds]**3 / 3
    inds = (d <= (r2 - r1))
    if inds.any():
        intersection[inds] = 4*r1[inds]**3 / 3

    union = sum_volume - intersection
    iou = intersection / (union + eps)
    assert ((iou >= 0) & (iou <= 1)).all()

    return iou

def point_in_sphere_mask(spheres, coords, soft=False, thr=5., eps=1e-6):
    """Return a mask indicate whether point in sphere. 
    Extend from https://arxiv.org/pdf/1906.01140.pdf.
    Args:
        spheres (Tensor): shape (N, 4)
        coords (Tensor): shape (P, 3)
        soft (bool): If True, use the soft mask
        thr (float): scale to normalize the distance

    Return:
        mask (Tensor): shape (N, P)
    """

    spheres = spheres[:, None, :]
    c = spheres[..., :3]
    r = spheres[..., 3]
    d = torch.norm(c - coords, p=2, dim=-1)
    if soft:
        # normalize the relative distancce such that [-r: r] -> [-thr: thr]
        rel_d = (r - d) * thr / (r + eps)
        mask = rel_d.sigmoid()
    else:
        mask = (r >= d).float()
    return mask

def soft_sphere_iou(spheres1, spheres2, coords, eps=1e-6):
    """Calculate the iou in term of how many point is in the intersection 
    between two sphere

    Args:
        spheres1 (Tensor): shape (N, 4), pred spheres
        spheres2 (Tensor): shape (N, 4), gt spheres 
        coords (Tensor): shape (P, 3)
    """
    spheres1 = spheres1.float()
    spheres2 = spheres2.float()
    coords = coords.float()
    assert spheres1.size(0) == spheres2.size(0)
    mask1 = point_in_sphere_mask(spheres1, coords, soft=True)
    mask2 = point_in_sphere_mask(spheres2, coords, soft=False)
    intersection = (mask1 * mask2).sum(-1)
    union = mask1.sum(-1) + mask2.sum(-1) - intersection
    iou = intersection / (union + eps)
    assert ((iou >= 0) & (iou <= 1)).all()
    return iou


def nms(dets, thresh):
    spheres = dets[:, :4]
    scores = dets[:, 4]

    # volumes = (x2 - x1) * (y2 - y1) * (z2 - z1)
    order = scores.argsort(descending=True)
    # order = scores.argsort()[::-1]

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        # xx1 = np.maximum(x1[i], x1[order[1:]])
        # yy1 = np.maximum(y1[i], y1[order[1:]])
        # zz1 = np.maximum(z1[i], z1[order[1:]])
        # xx2 = np.minimum(x2[i], x2[order[1:]])
        # yy2 = np.minimum(y2[i], y2[order[1:]])
        # zz2 = np.minimum(z2[i], z2[order[1:]])

        # w = np.maximum(0.0, xx2 - xx1)
        # h = np.maximum(0.0, yy2 - yy1)
        # d = np.maximum(0.0, zz2 - zz1)
        # inter = w * h * d
        # ovr = inter / (volumes[i] + volumes[order[1:]] - inter)
        iou = sphere_iou(spheres[i:i+1], spheres[order[1:]], aligned=False)

        inds = torch.nonzero(iou.squeeze(0) < thresh).squeeze(1)
        order = order[inds + 1]

    return dets[keep, :], keep


if __name__ == '__main__':
    # test case 1: r1 + r2 < d
    spheres1 = torch.tensor([[0, 0, 0, 3]]).float()
    spheres2 = torch.tensor([[0, 0, 2, 4]]).float()
    print(sphere_iou(spheres1, spheres2))

    # test case 2: r1 + r2 > d
    spheres1 = torch.tensor([[0, 0, 0, 3]]).float()
    spheres2 = torch.tensor([[0, 0, 9, 4]]).float()
    print(sphere_iou(spheres1, spheres2))


    # test case 3: one inside one other
    spheres1 = torch.tensor([[0, 0, 0, 3]]).float()
    spheres2 = torch.tensor([[0, 0, 0, 6]]).float()
    import pdb; pdb.set_trace()
    print(sphere_iou(spheres1, spheres2))



