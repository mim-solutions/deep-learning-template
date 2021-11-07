# import torch
import torch
import robbytorch as robby
from torchmetrics import Accuracy
from typing import Any, Dict

from hydra_lightning.shared.modules.base import BaseModule


class BaseAdvModule(BaseModule):
    """
    Base module for computing adversarial accuracy during validation_step.
    """

    def initialize_model(self):
        self.cross_entropy_criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy()
        self.std_val_accuracy = Accuracy()
        self.adv_val_accuracy = Accuracy()

    # for robby.input_transforms.PGD
    @staticmethod
    def forward_robby(module, dataitem: Any, phase: str) -> Dict[str, torch.Tensor]:
        targets = dataitem["target"]
        logits = module(dataitem["data"])
        loss = module.cross_entropy_criterion(logits, targets)
        return {'loss': loss}

    def compute_advs(self, dataitem, eps=1.25, step_size=0.5, Nsteps=20) -> torch.Tensor:
        with torch.enable_grad():
            return robby.input_transforms.PGD(
                self, dataitem, self.__class__.forward_robby,
                eps=eps, step_size=step_size, Nsteps=Nsteps, use_tqdm=False, minimize=False)

    def validation_step(self, dataitem: Any, batch_idx: int):
        targets = dataitem["target"]
        adv_logits = self(self.compute_advs(dataitem))
        adv_loss = self.cross_entropy_criterion(adv_logits, targets)

        std_logits = self(dataitem['data'])
        std_loss = self.cross_entropy_criterion(std_logits, targets)

        adv_acc = self.adv_val_accuracy(adv_logits, targets)
        self.log("val/adv_loss", adv_loss)
        self.log("val/adv_acc", adv_acc)

        std_acc = self.std_val_accuracy(std_logits, targets)
        self.log("val/std_loss", std_loss)
        self.log("val/std_acc", std_acc)

    def test_step(self, dataitem: Any, batch_idx: int):
        pass
