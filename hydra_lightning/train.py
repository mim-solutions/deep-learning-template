from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase


def load_from_config(config: DictConfig) -> Tuple[Trainer, LightningModule, LightningDataModule]:
    """
    Instantiates all PyTorch Lightning objects from config.
    This also calls seed_everything(config.seed, workers=True), hence it's convenient for reproducing results.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    trainer = load_trainer_from_config(config)
    module = load_module_from_config(config)
    datamodule = load_datamodule_from_config(config)

    return trainer, module, datamodule


def load_module_from_config(config: DictConfig) -> LightningModule:
    # Init lightning module
    print(f"Instantiating module <{config.module._target_}>")
    # _recursive_=False - never pass complex objects as module parameters (easier model serialization)
    module: LightningModule = hydra.utils.instantiate(config.module, _recursive_=False)
    return module


def load_trainer_from_config(config: DictConfig) -> Trainer:
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                lg = hydra.utils.instantiate(lg_conf)
                lg.log_hyperparams(config)
                logger.append(lg)

    # Init lightning trainer
    print(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    return trainer


def load_datamodule_from_config(config: DictConfig) -> LightningDataModule:
    # Init lightning datamodule
    print(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    return datamodule


def load_config(overrides: Optional[List] = None,
                config_path: str = "./configs", config_name: str = "config", job_name: str = "app") -> DictConfig:
    """
    Args:
        overrides: list of hydra command-line overrides, examples:
        - override specific key (use dots):
            overrides=["datamodule.transforms.transforms.0._target_=lib.utils.identity"]
        - delete key (doesn't work with lists):
            overrides=["~datamodule.transforms"])
        - override file (use slashes instead of dots!):
            overrides=["experiment=left_right_ovary/robust_backbone", "datamodule/transforms=default"])
        - add key:
            overrides=["+new_key=314"])
    """
    overrides = overrides or []
    with hydra.initialize(config_path=config_path, job_name=job_name):
        return hydra.compose(config_name=config_name, overrides=overrides)


def config_to_yaml(config: DictConfig) -> str:
    return OmegaConf.to_yaml(config, resolve=True)


def train(config: DictConfig):
    trainer, module, datamodule = load_from_config(config)
    print("Starting training!")
    trainer.fit(model=module, datamodule=datamodule)
