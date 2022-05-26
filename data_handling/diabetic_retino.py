from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from skimage.io import imread
from data_handling.preprocessing_scripts import run_diabetic_preprocessing_script

from default_paths import DATA_DIR_DIABETIC, DATA_DIR_DIABETIC_PROCESSED_IMAGES


class DiabeticRethinopathyDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, transform=transform)
        self.root = Path(self.root)  # type: ignore
        self.dataset_dataframe = dataframe
        self.targets = self.dataset_dataframe.level.apply(
            lambda x: int(x <= 1)
        ).values.astype(np.int64)
        self.file_ids = self.dataset_dataframe.image.values

    def __getitem__(self, index: int):
        filename = self.file_ids[index]
        target = self.targets[index]
        scan_image = imread(self.root / f"{filename}.jpeg").astype(np.float32)
        scan_image = self.transform(scan_image)
        return scan_image, target

    def __len__(self) -> int:
        return len(self.file_ids)


class DiabeticRetinopathyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms,
        val_transforms,
        batch_size,
        num_workers,
        shuffle,
        val_split=0.1,
    ):
        super().__init__()
        self.root_dir = DATA_DIR_DIABETIC
        self.extract_dir = DATA_DIR_DIABETIC_PROCESSED_IMAGES
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.shuffle = shuffle
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        if not DATA_DIR_DIABETIC_PROCESSED_IMAGES.exists():
            print(
                "Preprocessed data does not exist yet. Running preprocessing script. This may take a few hours."
            )
            assert (
                DATA_DIR_DIABETIC.exists()
            ), f"Data dir: {DATA_DIR_DIABETIC} does not exist. Have you updated default_paths.py?"
            run_diabetic_preprocessing_script()
        train_df = pd.read_csv(self.root_dir / "trainLabels.csv")
        val_test_df = pd.read_csv(self.root_dir / "retinopathy_solution.csv")
        val_df = val_test_df.loc[val_test_df.Usage == "Public"]
        test_df = val_test_df.loc[val_test_df.Usage == "Private"]
        self.dataset_train = DiabeticRethinopathyDataset(
            self.extract_dir / "train", train_df, self.train_transforms
        )
        self.dataset_val = DiabeticRethinopathyDataset(
            self.extract_dir / "test", val_df, self.val_transforms
        )
        self.dataset_test = DiabeticRethinopathyDataset(
            self.extract_dir / "test", test_df, self.val_transforms
        )

        print("#train: ", len(self.dataset_train))
        print("#val:   ", len(self.dataset_val))
        print("#test:  ", len(self.dataset_test))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
