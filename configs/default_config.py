from yacs.config import CfgNode
from pathlib import Path

config = CfgNode()

config.seed = None
config.run_name = "run"
config.dataset = None
config.n_classes = 10
config.model_name = None
config.training = CfgNode()
config.training.lr = 1e-3
config.training.weight_decay = 0.0
config.training.num_epochs = 200
config.training.batch_size = 128
config.training.p_dropout = 0.0
config.training.pretrained = False
config.training.milestones = None

config.augmentations = CfgNode()
config.augmentations.random_rotation = None
config.augmentations.horizontal_flip = False
config.augmentations.normalize = None
config.augmentations.resize = None
config.augmentations.center_crop = None
config.augmentations.expand_channels = False
config.augmentations.random_crop = None
config.augmentations.random_color_jitter = None

config.training.swag = CfgNode()
config.training.swag.swa_start = 5
config.training.swag.swa_lr = 0.05
config.training.swag.lr_init = 0.1
config.training.swag.momentum = 0.9
config.training.swag.num_epochs = 100

config.training.duq = CfgNode()
config.training.duq.lr = 0.05
config.training.duq.l_gradient_penalty = 0.5
config.training.duq.gamma = 0.999
config.training.duq.length_scale = 0.1
config.training.duq.weight_decay = 5e-4
config.training.duq.milestones = []


def load_yaml_training_config(config_path: Path) -> CfgNode:
    """
    Loads augmentations configs defined as yaml files.
    """
    yaml_config = config.clone()
    yaml_config.merge_from_file(config_path)
    return yaml_config
