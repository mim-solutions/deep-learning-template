import torch
import torchvision.transforms
import robbytorch as robby

from typing import Tuple
from collections import OrderedDict


def get_robust_backbone(arch: str, eps: float) -> Tuple[torch.nn.Module, int]:
    backbone = robby.models.get_model_from_robustness(arch, pretraining="microsoft", eps=eps)
    num_features = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    return torch.nn.Sequential(OrderedDict(
        normalization=torchvision.transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        backbone=backbone
    )), num_features
