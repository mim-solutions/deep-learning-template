# example contents of run.py file (should be at the root of the repo and in .gitignore)
from hydra_lightning.train import load_config, config_to_yaml, load_from_config


experiment = "example/standard_resnet_robust_train"
resume_from_checkpoint = False

epochs = 100 if resume_from_checkpoint else 60
wandb_run_id = '9d433aze' if resume_from_checkpoint else 'null'  # for robust_train
wandb_artifact_path = f"portal/dl_template_example/model-{wandb_run_id}:v0"  # for robust_train

cfg = load_config(overrides=[
    f"experiment={experiment}",
    "gpus=[6, 7]",
    f"epochs={epochs}",
    "task.run_name='test'",
    "trainer=ddp",  # for multi GPU
    f"logger.wandb.id={wandb_run_id}"  # to resume a run, last element of `run_path` in wandb
])
print(config_to_yaml(cfg))
trainer, module, datamodule = load_from_config(cfg)

if resume_from_checkpoint:
    import wandb

    api = wandb.Api()

    artifact = api.artifact(wandb_artifact_path, type="model")
    artifact_dir = artifact.checkout()
    ckpt_path = artifact.file()
    module = module.load_from_checkpoint(ckpt_path)

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
else:
    trainer.fit(model=module, datamodule=datamodule)
