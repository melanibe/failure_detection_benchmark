from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, List
from skimage.io import imread
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import numpy as np

from default_paths import DATA_BUSI


class BUSIDataset(VisionDataset):
    def __init__(
        self,
        filenames: List[Path],
        targets: np.ndarray,
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=".", transform=transform)
        self.filenames = filenames
        self.targets = targets

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        target = self.targets[index]
        scan_image = imread(filename, as_gray=True).astype(np.float32)
        scan_image = self.transform(scan_image)
        return scan_image, target

    def __len__(self) -> int:
        return len(self.filenames)


class BUSIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms,
        val_transforms,
        batch_size,
        num_workers,
        shuffle,
    ):
        super().__init__()
        self.root_dir = DATA_BUSI
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:

        normal_filenames = list((self.root_dir / "normal").glob("*).png"))
        benign_filenames = list((self.root_dir / "benign").glob("*).png"))
        malignant_filenames = list((self.root_dir / "malignant").glob("*).png"))

        train_val_normal, test_normal = train_test_split(
            normal_filenames, test_size=0.2, shuffle=False
        )
        train_normal, val_normal = train_test_split(
            train_val_normal, test_size=0.125, shuffle=False
        )
        y_train_normal, y_val_normal, y_test_normal = (
            np.repeat(0, len(train_normal)),
            np.repeat(0, len(val_normal)),
            np.repeat(0, len(test_normal)),
        )

        train_val_benign, test_benign = train_test_split(
            benign_filenames, test_size=0.2, shuffle=False
        )
        train_benign, val_benign = train_test_split(
            train_val_benign, test_size=0.125, shuffle=False
        )
        y_train_benign, y_val_benign, y_test_benign = (
            np.repeat(1, len(train_benign)),
            np.repeat(1, len(val_benign)),
            np.repeat(1, len(test_benign)),
        )

        train_val_malignant, test_malignant = train_test_split(
            malignant_filenames, test_size=0.2, shuffle=False
        )
        train_malignant, val_malignant = train_test_split(
            train_val_malignant, test_size=0.125, shuffle=False
        )
        y_train_malignant, y_val_malignant, y_test_malignant = (
            np.repeat(2, len(train_malignant)),
            np.repeat(2, len(val_malignant)),
            np.repeat(2, len(test_malignant)),
        )

        train_files = train_normal + train_benign + train_malignant
        y_train = np.concatenate((y_train_normal, y_train_benign, y_train_malignant))

        val_files = val_normal + val_benign + val_malignant
        y_val = np.concatenate((y_val_normal, y_val_benign, y_val_malignant))

        test_files = test_normal + test_benign + test_malignant
        y_test = np.concatenate((y_test_normal, y_test_benign, y_test_malignant))

        self.dataset_train = BUSIDataset(
            train_files,
            y_train,
            transform=self.train_transforms,
        )
        self.dataset_val = BUSIDataset(
            val_files,
            y_val,
            transform=self.val_transforms,
        )
        self.dataset_test = BUSIDataset(
            test_files,
            y_test,
            transform=self.val_transforms,
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
