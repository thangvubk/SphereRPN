import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scale):
        self.base_size = base_size
        self.scale = scale
        self.base_anchor = self.get_base_anchor()

    def get_base_anchor(self):
        # w = (self.base_size * 0.5) * self.scale
        # h = (self.base_size * 0.5) * self.scale
        # d = (self.base_size * 0.5) * self.scale
        # base_anchor = torch.tensor([-w, -h, -d, w, h, d])
        base_anchor = torch.tensor([0, 0, 0, self.base_size * self.scale])
        return base_anchor

    def get_volume(self):
        return (self.base_anchor[3:] - self.base_anchor[:3]).prod()

    def get_size(self):
        return self.base_anchor[3]

    def feat_coords_to_pc_coords(self, feat_coords, stride, pc_range, voxel_size):
        # x_pc = x_feat * stride * voxel_size + pc_range_min
        feat_coords = feat_coords.float()
        voxel_size = voxel_size
        pc_range_min = pc_range[:3]
        pc_coords = feat_coords * stride * voxel_size + pc_range_min
        return pc_coords

    def generate(self, feat_coords, stride): #, pc_range, voxel_size):
        device = feat_coords.device
        base_anchor = self.base_anchor.to(device)
        # pc_coords = self.feat_coords_to_pc_coords(feat_coords, stride, pc_range, voxel_size)
        pc_coords = feat_coords.float() * stride
        # anchor_xyz_start = pc_coords + base_anchor[:3]
        # anchor_xyz_end = pc_coords + base_anchor[3:]
        anchor_centroid = pc_coords + base_anchor[:3]
        anchor_radius = base_anchor[3:].expand(pc_coords.size(0), 1)
        anchors = torch.cat([anchor_centroid, anchor_radius], dim=-1)
        return anchors
