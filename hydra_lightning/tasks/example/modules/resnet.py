# import torch
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import Any

from lib.models.robust_backbone import get_robust_backbone
from .base_adv import BaseAdvModule


class ResnetModule(BaseAdvModule):

    def initialize_model(self):
        super().initialize_model()

        cfg: DictConfig = self.hparams.model_config
        self.backbone, num_features = get_robust_backbone(cfg.arch, cfg.eps)
        self.head = nn.Linear(num_features, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def training_step(self, dataitem: Any, batch_idx: int) -> torch.Tensor:
        targets = dataitem["target"]
        logits = self(dataitem["data"])
        loss = self.cross_entropy_criterion(logits, targets)

        acc = self.train_accuracy(logits, targets)
        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss
