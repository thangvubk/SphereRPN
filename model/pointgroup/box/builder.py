# from pcdet.utils import build_from_cfg

# from .registry import ASSIGNERS, SAMPLERS


def build_assigner(cfg):
    return build_from_cfg(cfg, ASSIGNERS)


def build_sampler(cfg):
    return build_from_cfg(cfg, SAMPLERS)
