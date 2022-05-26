from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from skimage.io import imread
from data_handling.preprocessing_scripts import run_rsna_preprocessing_script

from default_paths import DATA_DIR_RSNA, DATA_DIR_RSNA_PROCESSED_IMAGES


class RSNAPneumoniaDetectionDataModule(pl.LightningDataModule):
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
        self.root_dir = DATA_DIR_RSNA
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.shuffle = shuffle
        self.val_split = val_split
        if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
            print(
                "Preprocessed data does not exist yet. Running preprocessing script. This may take a few hours."
            )
            assert (
                DATA_DIR_RSNA.exists()
            ), f"Data dir: {DATA_DIR_RSNA} does not exist. Have you updated default_paths.py?"
            run_rsna_preprocessing_script()

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(self.root_dir / "stage_2_train_labels.csv")[
            ["patientId", "Target"]
        ].drop_duplicates()
        # Use 80% of dataset for train / val and 20% for test
        indices_train_val, indices_test = train_test_split(
            np.arange(len(train_df)), test_size=0.20, random_state=42
        )
        train_val_df = train_df.iloc[indices_train_val]
        test_df = train_df.iloc[indices_test]
        indices_train, indices_val = train_test_split(
            np.arange(len(train_val_df)), test_size=self.val_split, random_state=42
        )
        train_df = train_val_df.iloc[indices_train]
        val_df = train_val_df.iloc[indices_val]

        self.dataset_train = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=train_df,
            transform=self.train_transforms,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=val_df,
            transform=self.val_transforms,
        )
        self.dataset_test = RNSAPneumoniaDetectionDataset(
            str(DATA_DIR_RSNA_PROCESSED_IMAGES),
            dataframe=test_df,
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


class RNSAPneumoniaDetectionDataset(VisionDataset):
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
        self.targets = self.dataset_dataframe.Target.values.astype(np.int64)
        self.subject_ids = self.dataset_dataframe.patientId.values
        self.filenames = [
            self.root / f"{subject_id}.png" for subject_id in self.subject_ids
        ]

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        target = self.targets[index]
        scan_image = imread(filename).astype(np.float32)
        scan_image = self.transform(scan_image)
        return scan_image, target

    def __len__(self) -> int:
        return len(self.filenames)
