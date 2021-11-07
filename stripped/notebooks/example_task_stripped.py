#!/usr/bin/env python
# coding: utf-8

# # Example of training

# ## Imoports

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os

while 'notebooks' in os.getcwd():
    os.chdir("../")


# In[2]:


import robbytorch as robby

from hydra_lightning.train import load_from_config, load_config, config_to_yaml


# ## Choose experiment

# In[3]:


experiment = 'example/standard_resnet_robust_train'
# experiment = 'example/standard_resnet'

resume_from_checkpoint = False
epochs = 100 if resume_from_checkpoint else 60
wandb_run_id = '9d433aze' if resume_from_checkpoint else 'null'  # for robust_train
wandb_artifact_path = f"portal/dl_template_example/model-{wandb_run_id}:v0"  # for robust_train


# ## Load config

# In[4]:


cfg = load_config(overrides=[
    f"experiment={experiment}",
    "gpus=[6]",
    f"epochs={epochs}",
    f"logger.wandb.id={wandb_run_id}" # to resume a run, last element of `run_path` in wandb
])
print(config_to_yaml(cfg))
trainer, module, datamodule = load_from_config(cfg)


# ## Train

# In[5]:


if resume_from_checkpoint:
    import wandb
    artifact = wandb.use_artifact(wandb_artifact_path, type='model')
    artifact_dir = artifact.download()

    ckpt_path = artifact.file
    # module2 = module.load_from_checkpoint(artifact.file()) # loading module from checkpoint

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=artifact.file())
else:
    trainer.fit(model=module, datamodule=datamodule)


# # Visualize data

# In[6]:


dataitem = next(iter(datamodule.train_dataloader()))
batch = dataitem["data"]


# In[7]:


robby.get_image_table(batch[:5], size=(10,10))
pass


# In[ ]:




