import torch
import torchvision.transforms as tf
from torchvision.transforms.transforms import CenterCrop
from yacs.config import CfgNode


class ExpandChannels:
    """
    Transform 1-channel into 3-channel image, by copying the channel 3 times.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


def get_augmentations_from_config(config: CfgNode):
    """
    Return transformation pipeline as per config.
    """
    transform_list, val_transforms = [tf.ToTensor()], [tf.ToTensor()]
    if config.augmentations.random_crop is not None:
        transform_list.append(
            tf.RandomResizedCrop(
                config.augmentations.resize, scale=config.augmentations.random_crop
            )
        )
        val_transforms.append(tf.Resize(config.augmentations.resize))
    elif config.augmentations.resize is not None:
        transform_list.append(tf.Resize(config.augmentations.resize))
        val_transforms.append(tf.Resize(config.augmentations.resize))
    if config.augmentations.random_rotation is not None:
        transform_list.append(tf.RandomRotation(config.augmentations.random_rotation))
    if config.augmentations.horizontal_flip:
        transform_list.append(tf.RandomHorizontalFlip())
    if config.augmentations.random_color_jitter:
        transform_list.append(
            tf.ColorJitter(
                brightness=config.augmentations.random_color_jitter,
                contrast=config.augmentations.random_color_jitter,
            )
        )
    if config.augmentations.center_crop is not None:
        transform_list.append(CenterCrop(config.augmentations.center_crop))
        val_transforms.append(CenterCrop(config.augmentations.center_crop))
    # To artificially transform 1-channel to 3-channels by copying the value over the 3 channels.
    if config.augmentations.expand_channels:
        transform_list.append(ExpandChannels())
        val_transforms.append(ExpandChannels())
    return tf.Compose(transform_list), tf.Compose(val_transforms)
