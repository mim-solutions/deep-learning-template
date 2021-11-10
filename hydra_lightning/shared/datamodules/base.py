from typing import Optional, Callable, Dict

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from robbytorch.utils import TensorLike
from torchvision.transforms import Compose
import numpy as np


class BaseDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataloader_spec: DictConfig,
        val_dataloader_spec_override: Optional[Dict] = None,
        transform_factory: Callable[..., Optional[Callable[[np.ndarray], torch.Tensor]]] = lambda: None,
        augmentation_factory: Callable[..., Optional[Callable[[torch.Tensor], torch.Tensor]]] = lambda: None,
        additional_config: Optional[DictConfig] = None
    ):
        super().__init__()

        self.dataloader_spec = {**dataloader_spec}
        self.val_dataloader_spec = {**dataloader_spec, **(val_dataloader_spec_override or {})}
        self.transform_factory = transform_factory
        self.augmentation_factory = augmentation_factory
        self.additional_config = additional_config

        # You should add "train", "val", "test" datasets in setup().
        self.datasets: Dict[str, Dataset] = {}

    def get_transform(self, stage: str) -> Optional[Callable[[TensorLike], torch.Tensor]]:
        if stage == 'train':
            return Compose([self.transform_factory(), self.augmentation_factory()])
        else:
            return self.transform_factory()

    def prepare_data(self):
        """
        This method is for anything that must be done in the main process before forking
        subprocesses for distributed training. Tasks such as downloading, preprocessing or
        saving to disk are good candidates for this method. One thing to be wary of
        is that any state set here will not be carried over to the subprocesses in
        distributed training, so you should be careful not to set any state here.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        This method is for anything that must be done for each subprocess for distributed training.
        You should construct actual PyTorch Datasets and set any necessary states here.

        Good place to set self.datasets.
        """
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets['train'],
            shuffle=True,
            **self.dataloader_spec
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets['val'],
            shuffle=False,
            **self.val_dataloader_spec
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets['test'],
            shuffle=False,
            **self.val_dataloader_spec
        )
