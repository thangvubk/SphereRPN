'''
PointGroup
Written by Li Jiang
'''

import torch
import torch.nn as nn
import spconv
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops
from .backbone import UBlock, ResidualBlock
from .resnet import ResNet, FPN
from .rpn_head import RPNHead


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    return loss, log_vars


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)  # 7.5M params

        self.resnet = ResNet(18, 6, 32)
        self.fpn = FPN([32, 32, 32, 64, 128, 256], 256)
        anchor_scale = 0.05 * 50
        anchor_strides=[4, 8, 16, 32, 64]
        self.rpn_head = RPNHead(256, 256, anchor_scale, anchor_strides)
        print(sum(p.numel() for p in self.resnet.parameters()))
        self.unet_last_conv = spconv.SubMConv3d(m, 100, kernel_size=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(64),
            nn.ReLU()
        )

        #### semantic segmentation
        self.linear = nn.Linear(64, classes + 1) # bias(default): True

        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)
        self.init_weights()

    def init_weights(self):
        self.resnet.init_weights()
        self.fpn.init_weights()
        self.rpn_head.init_weights()


    def forward(self, input, input_map, gt_infos, coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        out = self.resnet(input)
        semantic_feats, fpn_feats = self.fpn(out)
        rpn_outs = self.rpn_head(fpn_feats)
        semantic_feats = self.output_layer(semantic_feats)
        semantic_feats = semantic_feats.features[input_map.long()]
        semantic_scores = self.linear(semantic_feats)
        return rpn_outs, semantic_scores

def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    weight = torch.ones(21).cuda()
    weight[[0, 1, 20]] = 0.1
    semantic_criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        coord_feats = batch['coord_feats'].cuda()
        feats = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coord_feats), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        rpn_outs, semantic_scores = model(input_, p2v_map, None, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        test_cfg = {
            'nms_pre': 1000,
            'nms_post': 1000,
            'max_num': 100,  # check 50, 100, 200, 300
            'nms_thr': 0.25,
            'score_thr': 0.5,
        }
        rpn_outs = rpn_outs + (None, test_cfg)
        spheres = model.rpn_head.get_boxes(*rpn_outs)
        spheres = spheres[0]
        masks = model.rpn_head.get_masks(spheres, coords_float)
        return masks.cpu().numpy(), spheres[:, 4].cpu().numpy(), semantic_scores.max(1)[1].cpu().numpy()

    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        coord_feats = batch['coord_feats'].cuda()
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        gt_infos = batch['gt_infos']
        for info in gt_infos:
            for k, v in info.items():
                info[k] = v.cuda()

        if cfg.use_coords:
            feats = torch.cat((feats, coord_feats), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        rpn_outs, semantic_scores = model(input_, p2v_map, gt_infos, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        rpn_outs = rpn_outs + ([t['gt_spheres'] for t in gt_infos], None, None)
        # feat_coords = [feat.indices for feat in out[0]]
        # temp = self.rpn_head.get_anchors(feat_coords, None)
        loss = model.rpn_head.loss(*rpn_outs)
        loss['loss_semantic'] = 0.2 * semantic_criterion(semantic_scores, labels)
        loss, log_vars = parse_losses(loss)
        return loss, None, log_vars, log_vars
        # loss = model.loss_fn(ret, gt_infos)
        loss = {}
        loss['loss_semantic'] = semantic_criterion(ret['semantic_scores'], labels)
        return loss['loss_semantic'], None, loss, loss

    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if (epoch > cfg.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss
        if(epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[3] * score_loss)

        return loss, loss_out, infos


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
