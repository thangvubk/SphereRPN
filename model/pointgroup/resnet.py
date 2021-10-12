import torch
import torch.nn as nn

#from ..registry import BACKBONES
from .utils import ConvModule, SparseMaxPool3D
from .utils import xavier_init


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, indice_key=None):
        super(BasicBlock, self).__init__()
        assert indice_key is not None
        conv1_type = 'subm' if stride == 1 else 'spconv'
        conv1_key = indice_key if stride == 1 else f'{indice_key}_{conv1_type}'
        self.conv1 = ConvModule(
            in_planes,
            planes,
            3,
            stride=stride,
            padding=1,
            conv_type=conv1_type,
            indice_key=conv1_key)
        self.conv2 = ConvModule(
            planes, planes, 3, conv_type='subm', indice_key=indice_key, act=None)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            assert torch.equal(identity.indices, out.indices)

        out.features += identity.features
        return out


class Bottleneck(nn.Module):
    pass


class ResNet(nn.Module):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels, base_channels):
        super().__init__()
        self.depth = depth
        assert self.depth in self.arch_settings
        self.inplanes = in_channels
        self.baseplanes = base_channels
        block, layers = self.arch_settings[self.depth]

        self.layer0 = ConvModule(self.inplanes, self.baseplanes, 3, conv_type='subm', indice_key='layer0')
        self.layer1 = ConvModule(
            self.baseplanes, self.baseplanes, 3, stride=2, padding=2, conv_type='spconv', indice_key='layer1_spconv')
        self.inplanes = self.baseplanes
        self.layer2 = self._make_layers(
            block, self.baseplanes, layers[0], stride=2, indice_key='layer2')
        self.layer3 = self._make_layers(
            block, self.baseplanes * 2**1, layers[1], stride=2, indice_key='layer3')
        self.layer4 = self._make_layers(
            block, self.baseplanes * 2**2, layers[2], stride=2, indice_key='layer4')
        self.layer5 = self._make_layers(
            block, self.baseplanes * 2**3, layers[3], stride=2, indice_key='layer5')

    def init_weights(self):
        pass  # included in default ConvModule init

    def _make_layers(self, block, planes, num_blocks, stride=1, indice_key=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparseMaxPool3D(3, stride, padding=1),
                ConvModule(
                    self.inplanes, planes * block.expansion, 1, act=None, indice_key=indice_key))
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, downsample=downsample, indice_key=indice_key))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, indice_key=indice_key))
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return (out0, out1, out2, out3, out4, out5)


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, semantic_lvl=0, fpn_lvls=range(2, 6), extra_fpn_lvl=True):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.semantic_lvl = semantic_lvl
        self.fpn_lvls = fpn_lvls
        self.extra_fpn_lvl = extra_fpn_lvl

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.td_convs = nn.ModuleList()

        # NOTE: the indice_key should map to the backbone
        for idx, in_channels in enumerate(self.in_channels):
            # reduce num channels for low-level layers to reduce computational cost
            if idx == 0:
                out_channels = self.out_channels // 4
            elif idx == 1:
                out_channels = self.out_channels // 2
            else:
                out_channels = self.out_channels


            out_channels = (self.out_channels // 2**(fpn_lvls[0] - idx) if idx < fpn_lvls[0]
                            else self.out_channels)

            key = f'layer{idx}'
            # lateral convs
            l_conv = ConvModule(in_channels, out_channels, 1, act=None, indice_key=key)
            self.lateral_convs.append(l_conv)
            # fpn convs
            if idx in fpn_lvls:
                fpn_conv = ConvModule(out_channels, out_channels, 3, act=None, indice_key=key)
                self.fpn_convs.append(fpn_conv)
            # top down convs
            if idx > 0:
                td_in_channels = (self.out_channels // 2**(fpn_lvls[0] - idx) if idx < fpn_lvls[0]
                                  else self.out_channels)
                td_out_channels = (self.out_channels // 2**(fpn_lvls[0] - idx + 1) if idx < (fpn_lvls[0] + 1)
                                  else self.out_channels)
                inv_key = f'layer{idx}_spconv'

                td_conv = ConvModule(
                    td_in_channels,
                    td_out_channels,
                    3,
                    conv_type='invconv',
                    indice_key=inv_key)
                self.td_convs.append(td_conv)

        if extra_fpn_lvl:
            self.fpn_convs.append(ConvModule(self.out_channels, self.out_channels, 3, stride=2, act=None, conv_type='spconv'))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvModule):
                xavier_init(m[0], distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # top-down path
        num_tds = len(self.td_convs)
        for i in list(range(num_tds))[::-1]:
            td_conv = self.td_convs[i]
            td_feat = td_conv(laterals[i + 1])
            laterals[i].features = laterals[i].features + td_feat.features

        # semantic feature
        semantic_feats = laterals[self.semantic_lvl]

        # fpn feats
        fpn_feats = []
        for i, lvl in enumerate(self.fpn_lvls):
            fpn_feats.append(self.fpn_convs[i](laterals[lvl]))
        if self.extra_fpn_lvl:
            fpn_feats.append(self.fpn_convs[-1](laterals[-1]))

        return semantic_feats, fpn_feats
