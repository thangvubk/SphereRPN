from functools import partial

import numpy as np
import torch
import torch.nn as nn
from .anchor_generator import AnchorGenerator
from .anchor_target import anchor_target, delta2sphere
from .utils import feats_at_batch, get_batch_ids, multi_apply

from .utils import ConvModule, SubMConv3D, normal_init
from .losses import IoULoss, L1Loss, CrossEntropyLoss, SoftIoULoss

INF = 1e10


class RPNHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 anchor_scale,
                 anchor_strides,
                 cls_loss=dict(type='FocalLoss', weight=1.),
                 box_loss=dict(type='IoULoss', weight=0.1)):
        super(RPNHead, self).__init__()

        self.channels = in_channels
        self.anchor_strides = anchor_strides
        self.anchor_scale = anchor_scale
        self.loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=1)
        self.loss_l1 = L1Loss(loss_weight=1)
        self.loss_iou = IoULoss(loss_weight=1)
        self.loss_soft_iou = SoftIoULoss(loss_weight=1)
        self.use_focal = False

        self.anchor_base_sizes = self.anchor_strides
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, self.anchor_scale))

        self.volume_ranges = self.get_volume_ranges(
            [ag.get_volume() for ag in self.anchor_generators])

        self._init_layers()

    def _init_layers(self):
        # NOTE donot use indice_key here since they are shared accross FPN levels
        self.rpn_conv = ConvModule(self.channels, self.channels, 3, indice_key=None)
        self.cls_conv = SubMConv3D(self.channels, 1, 1, indice_key=None)
        self.box_conv = SubMConv3D(self.channels, 4, 1, indice_key=None)

    def init_weights(self):
        normal_init(self.rpn_conv[0], std=0.01)  # index 0 is conv layer in the sequential
        # bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_conv, std=0.01)
        normal_init(self.box_conv, std=0.01)

    def forward_single(self, x):
        rpn_feat = self.rpn_conv(x)
        cls_score = self.cls_conv(rpn_feat)
        box_pred = self.box_conv(rpn_feat)
        return cls_score, box_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_volume_ranges(self, volumes):
        """Compute volume range.
        Args:
            - volumes (list): volume of each anchor level
        Returns:
            - volume_ranges (Tensor): shape (nlvl, 2), min an max range each level
        """
        num_levels = len(volumes)
        volume_ranges = [0]
        for i in range(num_levels - 1):
            pivot = (volumes[i] + volumes[i + 1]) / 2
            volume_ranges.append(pivot)
            volume_ranges.append(pivot)
        volume_ranges.append(INF)
        volume_ranges = np.array(volume_ranges)
        volume_ranges = volume_ranges.reshape(-1, 2)
        return volume_ranges

    def get_anchors_single(self, anchor_generator, feat_coords, stride, meta_infos=None):
        batch_ids = feat_coords[:, 0].unique()

        batch_anchors = []
        for i, batch_id in enumerate(batch_ids):
            mask = feat_coords[:, 0] == batch_id
            _feat_coords = feat_coords[mask, 1:]
            anchors = anchor_generator.generate(_feat_coords, stride) #, pc_range, voxel_size)
            batch_anchors.append(anchors)
        return batch_anchors

    def get_anchors(self, feat_coords, meta_infos):
        """
        Args:
            - feat_coords (list[Tensor]): each has shape (N, 4), columns are [batchid, x, y, z]
            - meta_infos (dict): infos need to generate anchors
        Returns:
            - anchor_list (list[list[Tensor]]): per level, per scan anchors
        """
        get_anchors_single_partial = partial(self.get_anchors_single, meta_infos=meta_infos)
        anchor_list = list(
            map(get_anchors_single_partial, self.anchor_generators, feat_coords,
                self.anchor_strides))
        return anchor_list

    def loss(self, cls_scores, box_preds, gt_spheres, meta_infos, cfg):
        # gt_boxes = scan_infos['boxes']
        gt_boxes = gt_spheres
        feat_coords = [feat.indices for feat in cls_scores]
        
        anchors = self.get_anchors(feat_coords, meta_infos)
        regression_ranges = [(-1, 20.0), (20.0, 40.0), (40.0, 80.0), (80, 160), (160, INF)]
        anchor_targets = anchor_target(anchors, gt_boxes, regression_ranges, cfg)
        (flat_anchors, flat_label_targets, flat_label_weights, flat_box_targets, flat_box_weights,
         num_pos, num_neg) = anchor_targets
        num_samples = num_pos if self.use_focal else num_pos + num_neg

        # flattten multi-level features
        flat_cls_scores = torch.cat([stensor.features.squeeze() for stensor in cls_scores], dim=0)
        flat_box_preds = torch.cat([stensor.features for stensor in box_preds], dim=0)

        loss_cls = self.loss_cls(
            flat_cls_scores, flat_label_targets, weight=flat_label_weights, avg_factor=num_samples)


        loss_l1 = self.loss_l1(
            flat_box_preds, flat_box_targets, weight=flat_box_weights, avg_factor=num_samples)
        decoded_box_preds = delta2sphere(flat_anchors, flat_box_preds)
        decoded_box_targets = delta2sphere(flat_anchors, flat_box_targets)
        loss_iou = self.loss_iou(decoded_box_preds, decoded_box_targets, weight=flat_box_weights, avg_factor=num_samples)
        # compute soft iou loss on each scan
        flat_pred_batch_ids = torch.cat([f[:, 0] for f in feat_coords])
        # we use point coords from feature coords for computational efficiency
        pc_coords = feat_coords[0]
        pc_coords[:, 1:] = pc_coords[:, 1:] * self.anchor_strides[0]
        batch_ids = flat_pred_batch_ids.unique()
        loss_soft_iou = []
        for batch_id in batch_ids:
            cur_pc_coords_inds = pc_coords[:, 0] == batch_id
            cur_pc_coords = pc_coords[cur_pc_coords_inds, 1:]
            cur_pred_batch_inds = flat_pred_batch_ids == batch_id
            # take pos box which have weight > 0 only for computational efficiency
            pos_inds = (flat_box_weights[:, 0] > 0)
            box_inds = (cur_pred_batch_inds & pos_inds)
            cur_decoded_box_pred = decoded_box_preds[box_inds]
            cur_decoded_box_targets = decoded_box_targets[box_inds]
            if box_inds.any():
                loss_soft_iou.append(self.loss_soft_iou(cur_decoded_box_pred,
                                                        cur_decoded_box_targets,
                                                        coords=cur_pc_coords,
                                                        avg_factor=num_samples))
            else:
                loss_soft_iou.append(cur_decoded_box_pred.sum())
        loss_soft_iou = sum(loss_soft_iou)
        
        return dict(loss_cls=loss_cls, loss_l1=loss_l1, loss_iou=loss_iou, loss_soft_iou=loss_soft_iou)

    def get_boxes(self, cls_scores, box_preds, meta_infos, cfg):
        feat_coords = [feat.indices for feat in cls_scores]

        anchors = self.get_anchors(feat_coords, meta_infos)
        # transpose to per image, per level
        anchors = list(zip(*anchors))

        proposal_list = []
        with torch.no_grad():
            batch_ids = get_batch_ids(cls_scores[0])
            for i, batch_id in enumerate(batch_ids):
                mlvl_cls_scores = [feats_at_batch(cls_score, batch_id) for cls_score in cls_scores]
                mlvl_box_preds = [feats_at_batch(box_pred, batch_id) for box_pred in box_preds]
                mlvl_anchors = anchors[i]
                proposals = self.get_boxes_single(mlvl_cls_scores, mlvl_box_preds, mlvl_anchors,
                                                  cfg)
                proposal_list.append(proposals)
        return proposal_list

    def get_boxes_single(self, mlvl_cls_scores, mlvl_box_preds, mlvl_anchors, cfg):
        nms_pre = cfg['nms_pre']
        nms_post = cfg['nms_post']
        max_num = cfg['max_num']
        nms_thr = cfg['nms_thr']
        mlvl_proposals = []
        for cls_scores, box_preds, anchors in zip(mlvl_cls_scores, mlvl_box_preds, mlvl_anchors):
            scores = cls_scores.sigmoid().reshape(-1)
            if nms_pre > 0 and scores.size(0) > nms_pre:
                _, topk_inds = scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                box_preds = box_preds[topk_inds, :]
                scores = scores[topk_inds]
            # import pdb; pdb.set_trace()
            proposals = delta2sphere(anchors, box_preds)
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            from .box import nms
            proposals, _ = nms(proposals, nms_thr)
            proposals = proposals[:nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, dim=0)
        scores = proposals[:, 4]
        num = min(max_num, proposals.shape[0])
        _, topk_inds = scores.topk(num)
        proposals = proposals[topk_inds, :]
        return proposals

    def get_masks(self, proposals, coords):
        from .box.sphere_iou import point_in_sphere_mask
        proposals = proposals[:, :4]
        mask = point_in_sphere_mask(proposals, coords)
        return mask.byte()
