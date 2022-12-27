# import torch
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from typing import Any

from lib.models.robust_backbone import get_robust_backbone
from .base_adv import BaseAdvModule


class ResnetRobustTrainModule(BaseAdvModule):

    def initialize_model(self):
        super().initialize_model()

        cfg: DictConfig = self.hparams.model_config
        self.adv_train_accuracy = Accuracy(task='multiclass', num_classes=cfg.num_classes)

        self.backbone, num_features = get_robust_backbone(cfg.arch, cfg.eps)
        self.head = nn.Linear(num_features, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def training_step(self, dataitem: Any, batch_idx: int) -> torch.Tensor:
        targets = dataitem["target"]
        adv_logits = self(self.compute_advs(dataitem))
        adv_loss = self.cross_entropy_criterion(adv_logits, targets)

        adv_acc = self.adv_train_accuracy(adv_logits, targets)
        self.log("train/adv_loss", adv_loss)
        self.log("train/adv_acc", adv_acc)

        with torch.no_grad():
            logits = self(dataitem['data'])
            loss = self.cross_entropy_criterion(logits, targets)

        acc = self.train_accuracy(logits, targets)
        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return adv_loss
