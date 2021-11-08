
from hydra_lightning.train import load_config, config_to_yaml, load_from_config


cfg = load_config(overrides=[
    "experiment=example/standard_resnet_robust_train",
    "gpus=[6, 7]",
    "task.run_name='test'",
    "trainer=ddp"  # for multi GPU
])
print(config_to_yaml(cfg))

trainer, module, datamodule = load_from_config(cfg)
trainer.fit(model=module, datamodule=datamodule)
