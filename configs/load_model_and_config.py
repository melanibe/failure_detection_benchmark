import os
from pathlib import Path
from typing import List

from data_handling.augmentations import get_augmentations_from_config
from data_handling.medmnist_modules import (
    OrganAMNISTModule,
    PathMNISTModule,
    TissueMNISTModule,
)
from data_handling.rsna_pneumonia import RSNAPneumoniaDetectionDataModule
from data_handling.diabetic_retino import DiabeticRetinopathyDataModule
from models import resnet, densenet_with_dropout
from data_handling.busi import BUSIDataModule
from pytorch_lightning.utilities.seed import seed_everything
from yacs.config import CfgNode

from configs.default_config import load_yaml_training_config


_model_name_to_module_cls = {
    "resnet18": resnet.ResNet18Module,
    "resnet50": resnet.ResNet50Module,
    "densenet": densenet_with_dropout.DenseNetModule,
    "wideresnet50": resnet.WideResNet50Module,
}

_dataset_name_to_module_cls = {
    "PathMNIST": PathMNISTModule,
    "TissueMNIST": TissueMNISTModule,
    "RSNAPneumonia": RSNAPneumoniaDetectionDataModule,
    "OrganAMNIST": OrganAMNISTModule,
    "BUSI": BUSIDataModule,
    "DiabeticRetino": DiabeticRetinopathyDataModule,
}


def get_modules(config, shuffle_training: bool = True):
    module_cls = _model_name_to_module_cls[config.model_name]
    train_transforms, val_transforms = get_augmentations_from_config(config)
    module = module_cls(
        config.n_classes,
        weight_decay=config.training.weight_decay,
        lr=config.training.lr,
        p_dropout=config.training.p_dropout,
        pretrained=config.training.pretrained,
        milestones=config.training.milestones,
    )
    data_module_cls = _dataset_name_to_module_cls[config.dataset]
    data_module = data_module_cls(
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=shuffle_training,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )
    return data_module, module


def get_data_module(config_name):
    config = load_yaml_training_config(Path(__file__).parent / config_name)
    train_transforms, val_transforms = get_augmentations_from_config(config)
    data_module_cls = _dataset_name_to_module_cls[config.dataset]
    data_module = data_module_cls(
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=True,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def get_output_dir_from_config(config: CfgNode) -> Path:
    return (
        Path(os.getenv("OUTPUT_DIR", "outputs"))
        / config.dataset
        / config.model_name
        / f"{config.run_name}"
        / f"seed_{config.seed}"
    )


def load_model_from_checkpoint(config, model_module):
    model = model_module.load_from_checkpoint(
        Path(config.output_dir) / "best.ckpt"
    ).model.eval()
    return model


def get_config_data_model_for_eval(config_name: str):
    config = load_yaml_training_config(Path(__file__).parent / config_name)
    data_module, model_module = get_modules(config, shuffle_training=False)
    data_module.train_transforms = data_module.val_transforms
    data_module.test_transforms = data_module.val_transforms
    data_module.prepare_data()
    data_module.setup()
    models = []
    output_dirs = []
    all_seeds = config.seed if isinstance(config.seed, List) else [config.seed]
    seed_everything(all_seeds[0], workers=True)
    for seed in all_seeds:
        config.seed = seed
        config.output_dir = str(get_output_dir_from_config(config))
        models.append(load_model_from_checkpoint(config, model_module))
        output_dirs.append(config.output_dir)
    return config, data_module, models, output_dirs


def get_failure_detection_filename(output_dir):
    folder_name = Path(output_dir) / "failure_detection"
    folder_name.mkdir(parents=True, exist_ok=True)
    return folder_name / "scores_df.csv"
