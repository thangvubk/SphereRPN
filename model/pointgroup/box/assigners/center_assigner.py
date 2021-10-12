import torch

# from ..registry import ASSIGNERS
# from ..transforms import get_sphere_centers, get_centers_inside_spheres
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

INF = 1e10


def get_centers_inside_spheres(centers, spheres):
    """
    Args:
        - centers (Tensor): shape (NA, NB, 3)
        - spheres (Tensor): shape (NA, NB, 6)
    Returns:
        - mask: shape (NA, NB), indicates whether centers is insdie spheres
    """
    s_centers = spheres[..., :3]
    s_radius = spheres[..., 3]
    distance = torch.norm(s_centers - centers, dim=-1)
    return distance < s_radius


class CenterAssigner(BaseAssigner):
    """Assign a corresponding gt sphere or background to each sphere

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def assign(self, spheres, gt_spheres, radius_ranges, gt_labels=None):
        """ Assign gt_spheres to spheres if sphere_center is in gt_spheres.
        If two or more gt_spheres contain sphere_center, the smallest gt_spheres will be used.

        Args:
            - spheres (Tensor): bounding spheres to be assigned, shape (n, 6)
            - gt_spheres (Tensor): groundtruth spheres, shape (k, 6)
            - radius_range (Tensor): range to map gt_spheres to feat level
            - gt_labels (Tensor): label of gt_spheres, shape (k, )
        """
        spheres = spheres[:, :4]
        num_spheres = spheres.size(0)
        num_gts = gt_spheres.size(0)
        if num_gts == 0:
            assigned_gt_inds = spheres.new_zeros((num_spheres, )).long()
            if gt_labels is not None:
                assigned_labels = spheres.new_zeros((num_spheres, )).long()
            else:
                assigned_labels = None
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # radiuses = (gt_spheres[:, 3:] - gt_spheres[:, :3]).prod(dim=-1)
        radiuses = gt_spheres[:, 3]
        # radiuses = radiuses[None].repeat(num_spheres, 1)
        radiuses = radiuses[None].repeat(num_spheres, 1)
        # radius_ranges = radius_ranges[:, None, :].expand(num_spheres, num_gts, 2)
        radius_ranges = radius_ranges[:, None, :].expand(num_spheres, num_gts, 2)
        # sphere_centers = get_sphere_centers(spheres)

        sphere_centers = spheres[:, :3]
        sphere_centers = sphere_centers[:, None, :].expand(num_spheres, num_gts, 3)
        spheres = spheres[:, None, :].expand(num_spheres, num_gts, 4)
        gt_spheres = gt_spheres[None].expand(num_spheres, num_gts, 4)

        # Each sphere take the gt_spheres with min radius to be assigned index.
        # Boxes with center outside of gt_sphere or the radius doesnot match radius level -> negative.
        centers_inside_gt_spheres_mask = get_centers_inside_spheres(sphere_centers, gt_spheres)
        inside_radius_range = (radiuses >= radius_ranges[..., 0]) & (radiuses < radius_ranges[..., 1])
        radiuses[centers_inside_gt_spheres_mask == 0] = INF
        radiuses[inside_radius_range == 0] = INF
        min_radius, min_radius_inds = radiuses.min(dim=1)

        assigned_gt_inds = min_radius_inds + 1
        assigned_gt_inds[min_radius == INF] = 0

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_spheres, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze(1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


