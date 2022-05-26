import medmnist
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from default_paths import DATA_MEDMNIST


class MedMNISTModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        batch_size=32,
        num_workers=12,
        shuffle=True,
        *args,
        **kwargs
    ):
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=val_transforms,
        )
        self.root_dir = DATA_MEDMNIST
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    @property
    def medmnist_class_name(self):
        NotImplementedError

    def setup(self, stage=None) -> None:
        DataClass = getattr(
            medmnist, medmnist.INFO[self.medmnist_class_name]["python_class"]
        )
        self.dataset_train = DataClass(
            root=self.root_dir,
            split="train",
            transform=self.train_transforms,
            download=True,
        )
        self.dataset_test = DataClass(
            root=self.root_dir,
            split="test",
            transform=self.test_transforms,
            download=True,
        )
        self.dataset_val = DataClass(
            root=self.root_dir,
            split="val",
            transform=self.val_transforms,
            download=True,
        )
        self.dataset_train.labels = self.dataset_train.labels.reshape(-1)
        self.dataset_test.labels = self.dataset_test.labels.reshape(-1)
        self.dataset_val.labels = self.dataset_val.labels.reshape(-1)
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


class PathMNISTModule(MedMNISTModule):
    @property
    def medmnist_class_name(self):
        return "pathmnist"


class TissueMNISTModule(MedMNISTModule):
    @property
    def medmnist_class_name(self):
        return "tissuemnist"


class OrganAMNISTModule(MedMNISTModule):
    @property
    def medmnist_class_name(self):
        return "organamnist"
