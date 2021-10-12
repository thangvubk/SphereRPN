import torch
import torch.nn as nn
import numpy as np
from functools import partial 


class SparseTensorWrapper(object):

    def __init__(self, stensor):
        self.stensor = stensor
        self.coords = stensor.indices
        self.feats = stensor.features

    def sparse2batch(self):
        """Convert sparse tensor to batched tensor using as transformer input"""
        batch_ids = self.coords[:, 0].unique()
        N = batch_ids.size(0)
        C = self.feats.size(1)
        max_len = max([(self.coords[:, 0] == id).sum().item() for id in batch_ids])
        out_feats = self.feats.new_zeros((N, max_len, C))
        pads = self.feats.new_ones((N, max_len), dtype=torch.bool)
        for batch_id in batch_ids:
            mask = self.coords[:, 0] == batch_id
            cur_len = mask.sum()
            out_feats[batch_id, :cur_len, :] = self.feats[mask]
            pads[batch_id, :cur_len] = 0
        return out_feats, pads

    def batch2sparse(self, updated_feats=None):
        """Convert batched tensor to sparse tensor"""
        if updated_feats is None:
            return self.stensor
        updated_feats = updated_feats.permute(1, 0, 2)
        batch_ids = self.coords[:, 0].unique()
        lens = [(self.coords[:, 0] == id).sum().item() for id in batch_ids]
        updated_feats = [updated_feats[i, :ln] for i, ln in enumerate(lens)]
        updated_feats = torch.cat(updated_feats)
        assert updated_feats.size(0) == self.coords.size(0), 'features should have same first dimension with coords'
        self.stensor.features = updated_feats
        return self.stensor

def list2batch(tensor_list):
    """list of tensor shape [Ni, NPi] to [N, NP] where N = sum(Ni), NP = max(NPi)"""
    max_len = max([t.size(1) for t in tensor_list])
    out_tensors = []
    for i, t in enumerate(tensor_list):
        Ni = t.size(0)
        cur_len = t.size(1)
        tensor = t.new_zeros((Ni, max_len))
        tensor[:, :cur_len] = t
        out_tensors.append(tensor)
    out_tensors = torch.cat(out_tensors, dim=0)
    return out_tensors


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

import spconv


class SparseConv(spconv.conv.SparseConvolution):

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != [1] * len(self.stride):
            s += ', stride={stride}'
        if self.padding != [0] * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != [1] * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != [0] * len(self.output_padding):
            s += ', output_padding={output_padding}'
        s += ', indice_key={indice_key}'
        return s.format(**self.__dict__)


class SparseMaxPool3D(spconv.SparseMaxPool3d):

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, stride={stride}'
        if self.padding != [0] * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != [1] * len(self.dilation):
            s += ', dilation={dilation}'
        return s.format(**self.__dict__)


class SparseConv3D(SparseConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super(SparseConv3D, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)


class SparseInverseConv3D(SparseConv):

    def __init__(self, in_channels, out_channels, kernel_size, indice_key, bias=True):
        super(SparseInverseConv3D, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key)


class SubMConv3D(SparseConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super(SubMConv3D, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key)


def build_conv(conv_type,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               bias=True,
               indice_key=None):
    assert conv_type in ['subm', 'spconv', 'invconv']
    if conv_type == 'subm':
        return SubMConv3D(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
    elif conv_type == 'spconv':
        return SparseConv3D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            indice_key=indice_key)
    else:  # 'invconv'
        return SparseInverseConv3D(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=bias)


class ConvModule(spconv.SparseSequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 indice_key=None,
                 conv_type='subm',
                 bias='auto',
                 norm='bn',
                 act='relu',
                 inplace=True):
        assert conv_type in ['subm', 'spconv', 'invconv']
        assert bias in ['auto', False, True]
        assert norm in ['bn', None]
        assert act in ['relu', None]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.indice_key = indice_key
        self.conv_type = conv_type
        self.bias = bias
        self.norm = norm
        self.act = act
        self.inplace = inplace
        self.with_norm = norm is not None
        self.with_act = act is not None
        if self.bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        module = []
        conv = build_conv(
            conv_type,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            indice_key=indice_key)
        module.append(conv)

        if self.with_norm:
            norm = nn.BatchNorm1d(out_channels)  # TODO check momentum 0.01
            module.append(norm)

        if self.with_act:
            act = nn.ReLU(inplace=inplace)
            module.append(act)

        super(ConvModule, self).__init__(*module)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self[0])
        if self.with_norm:
            constant_init(self[1], 1, bias=0)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)




def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def multi_apply(func, *args, **kwargs):
    """Apply func on list of args with kwargs being shared.
    Args:
        - *args: each element in args should be a list
        - **kwargs: shared elements to apply func
    Returns:
        tupple(list)
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def read_txt(path):
    """Read txt file into lines."""
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def to_torch(data):
    """assume data is structures of list and/or dict. Note: this is inplace function"""

    def _dict_parser(data, device):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = _dict_parser(v, device)
            elif isinstance(v, list):
                data[k] = _list_parser(v, device)
            elif isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            else:
                pass  # keep other untouch
        return data

    def _list_parser(data, device):
        for i, item in enumerate(data):
            if isinstance(item, list):
                data[i] = _list_parser(item, device)
            elif isinstance(item, dict):
                data[i] = _dict_parser(item, device)
            elif isinstance(item, np.ndarray):
                data[i] = torch.from_numpy(item).to(device)
            elif isinstance(item, torch.Tensor):
                data[i] = item.to(device)
            else:
                pass  # keep other untouch
        return data

    assert isinstance(data, dict)
    device = torch.cuda.current_device()
    data = _dict_parser(data, device)

    return data


def get_batch_ids(stensor):
    """Get batch ids of sparse tensor"""
    return stensor.indices[:, 0].unique()


def feats_at_batch(stensor, batch_id):
    """get features of sparse tensor at batch id
    Args:
        - tensor (Sparse Tensor)
        - batch_id (int)
    Returns:
        - (Tensor)
    """
    mask = stensor.indices[:, 0] == batch_id
    return stensor.features[mask]
