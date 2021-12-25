from typing import Any, Callable, Optional, TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf
if TYPE_CHECKING:
    LightningModule = Any
else:
    from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from robbytorch.train import get_optimizer, get_scheduler


class BaseModule(LightningModule):
    forward: Callable[..., Any]  # used for inference (val/test)
    training_step: Callable[..., STEP_OUTPUT]
    validation_step: Callable[..., Optional[STEP_OUTPUT]]
    test_step: Callable[..., Optional[STEP_OUTPUT]]

    def __init__(self,
                 model_config: DictConfig,
                 optimizer_spec: DictConfig,
                 scheduler_spec: Optional[DictConfig] = None):
        super().__init__()
        """The params to this method will be stored in self.hparams"""

        # This line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        # Note that this saves just BaseModule#__init__ params;
        # If __init__ is overriden then you need to call self.save_hyperparameters() again
        # For the implementation details check pytorch_lightning.utilities.parsing.get_init_args
        self.save_hyperparameters(logger=False)
        self.initialize_model()

    def initialize_model(self):
        """
        You should instantiate self.model and other attributes here using self.hparams.model_config
        """
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = get_optimizer(OmegaConf.to_container(self.hparams.optimizer_spec), self)
        scheduler_spec = self.hparams.scheduler_spec
        if not scheduler_spec:
            return optimizer

        scheduler = get_scheduler(OmegaConf.to_container(scheduler_spec), optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
