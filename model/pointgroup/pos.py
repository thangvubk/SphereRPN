import torch
import math
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    Extend from https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    to work on 3D sparse tensor
    """  # noqa
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, stensor):
        coords = stensor.indices
        device = coords.device
        shape = stensor.spatial_shape
        x_embed = torch.arange(shape[0], dtype=torch.float32, device=device)
        y_embed = torch.arange(shape[1], dtype=torch.float32, device=device)
        z_embed = torch.arange(shape[2], dtype=torch.float32, device=device)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[-1] + eps) * self.scale
            y_embed = y_embed / (y_embed[-1] + eps) * self.scale
            z_embed = z_embed / (z_embed[-1] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        x_embed = x_embed[:, None, None].expand(shape)
        y_embed = y_embed[None, :, None].expand(shape)
        z_embed = z_embed[None, None, :].expand(shape)

        batch_ids = coords[:, 0].unique()
        N = batch_ids.size(0)
        max_len = max([(coords[:, 0] == id).sum().item() for id in batch_ids])

        pos_x = torch.zeros((N, max_len, self.num_pos_feats), device=device)
        pos_y = torch.zeros((N, max_len, self.num_pos_feats), device=device)
        pos_z = torch.zeros((N, max_len, self.num_pos_feats), device=device)
        for batch_id in batch_ids:
            mask = coords[:, 0] == batch_id
            cur_len = mask.sum()
            cur_coords = coords[mask].long()
            xx, yy, zz = cur_coords[:, 1], cur_coords[:, 2], cur_coords[:, 3]
            cur_x_embed = x_embed[xx, yy, zz]
            cur_y_embed = y_embed[xx, yy, zz]
            cur_z_embed = z_embed[xx, yy, zz]

            cur_pos_x = cur_x_embed[:, None] / dim_t
            cur_pos_y = cur_y_embed[:, None] / dim_t
            cur_pos_z = cur_z_embed[:, None] / dim_t

            cur_pos_x = torch.stack([cur_pos_x[:, 0::2].sin(), cur_pos_x[:, 1::2].cos()], dim=2).flatten(1)
            cur_pos_y = torch.stack([cur_pos_y[:, 0::2].sin(), cur_pos_y[:, 1::2].cos()], dim=2).flatten(1)
            cur_pos_z = torch.stack([cur_pos_z[:, 0::2].sin(), cur_pos_z[:, 1::2].cos()], dim=2).flatten(1)

            pos_x[batch_id, :cur_len, :] = cur_pos_x
            pos_y[batch_id, :cur_len, :] = cur_pos_y
            pos_z[batch_id, :cur_len, :] = cur_pos_z
        pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)
        return pos

