import torch
import pandas as pd
import numpy as np
import pathlib
import skimage.io

from typing import Callable, Dict, Any

from lib.utils import PathLike


class ExampleDataset(torch.utils.data.Dataset):

    def __init__(self, data_root: PathLike,
                 metadata: pd.DataFrame,
                 transform: Callable[[np.ndarray], torch.Tensor]):
        self.data_root = pathlib.Path(data_root).expanduser()
        self.metadata = metadata
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.metadata.iloc[idx]
        img_path = self.data_root / f"{record['label_name']}/{record['file_name']}"

        img_ndarray = skimage.io.imread(img_path)
        img_tensor = self.transform(img_ndarray)

        target = torch.tensor(record['label'])

        return {'data': img_tensor, 'target': target}

    def __len__(self):
        return len(self.metadata)
