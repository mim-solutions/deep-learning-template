from torchvision.transforms import Compose
from typing import Dict, Optional
from torch.utils.data import Dataset

import robbytorch as robby

from hydra_lightning.shared.datamodules.base import BaseDataModule
from lib.toml import load_toml_config
from lib.utils import get_root_dir_of_repo
from lib.datasets.example_dataset import ExampleDataset


class DefaultDataModule(BaseDataModule):

    def prepare_data(self):
        # Put expensive operations here; Don't set state here. Por. BaseDataModule
        pass

    def setup(self, stage: Optional[str] = None):
        # NOTE - we use config_example.toml but in your DataModule you should just call:
        # data_root = load_toml_config()['your_data_absolute_path']
        repo_root = get_root_dir_of_repo()
        config_example = repo_root / 'config_example.toml'
        data_relative_path = load_toml_config(toml_path=config_example)['example_data_relative_path']
        data_root = repo_root / data_relative_path

        metadata = robby.tutorial.load_cute_dataset(root=data_root)

        # NOTE - it's a toy example; we don't care if the split is deterministic.
        # Usually you should save the train_test_split as a csv.
        train_test_split = robby.datasets.split_proportionally(
            metadata, by='label', proportions={"train": 0.7, "val": 0.25, "test": 0.05})

        self.datasets: Dict[str, Dataset] = {}
        self.datasets['train'] = ExampleDataset(data_root, train_test_split['train'],
                                                transform=Compose(
                                                    [self.transform_factory(), self.augmentation_factory()])
                                                )

        for stage in ['val', 'test']:
            # We don't use augmentation here.
            self.datasets[stage] = ExampleDataset(data_root, train_test_split[stage],
                                                  transform=self.transform_factory())
