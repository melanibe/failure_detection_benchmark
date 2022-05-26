import cv2
import numpy as np


# All the preprocessing functions below adapted from
# https://github.com/tensorflow/datasets/blob/a456b49f1993f26a43d5618094b93ef0c0ea969f/tensorflow_datasets/image_classification/diabetic_retinopathy_detection.py#L231

# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _btgraham_processing(filepath, extract_dir, target_pixels=300, crop_to_radius=True):
    """Process an image as the winner of the 2015 Kaggle competition.
    Args:
      image_fobj: File object containing the original image.
      filepath: Filepath of the image, for logging purposes only.
      target_pixels: The number of target pixels for the radius of the image.
      crop_to_radius: If True, crop the borders of the image to remove gray areas.
    Returns:
      A file object.
    """
    # Decode image using OpenCV2.
    image = cv2.imread(str(filepath), flags=3)
    # Process the image.
    image = _scale_radius_size(image, filepath, target_radius_size=target_pixels)
    image = _subtract_local_average(image, target_radius_size=target_pixels)
    image = _mask_and_crop_to_radius(
        image,
        target_radius_size=target_pixels,
        radius_mask_ratio=0.9,
        crop_to_radius=crop_to_radius,
    )
    # Encode the image with quality=72 and store it to disk
    cv2.imwrite(
        str(extract_dir / f"{filepath.stem}.jpeg"),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 72],
    )


def _scale_radius_size(image, filepath, target_radius_size):
    """Scale the input image so that the radius of the eyeball is the given."""
    x = image[image.shape[0] // 2, :, :].sum(axis=1)
    r = (x > x.mean() / 10.0).sum() / 2.0
    if r < 1.0:
        # Some images in the dataset are corrupted, causing the radius heuristic to
        # fail. In these cases, just assume that the radius is the height of the
        # original image.
        r = image.shape[0] / 2.0
    s = target_radius_size / r
    return cv2.resize(image, dsize=None, fx=s, fy=s)


def _subtract_local_average(image, target_radius_size):
    image_blurred = cv2.GaussianBlur(image, (0, 0), target_radius_size / 30)
    image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
    return image


def _mask_and_crop_to_radius(
    image, target_radius_size, radius_mask_ratio=0.9, crop_to_radius=False
):
    """Mask and crop image to the given radius ratio."""
    mask = np.zeros(image.shape)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = int(target_radius_size * radius_mask_ratio)
    cv2.circle(mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1)
    image = image * mask + (1 - mask) * 128
    if crop_to_radius:
        x_max = min(image.shape[1] // 2 + radius, image.shape[1])
        x_min = max(image.shape[1] // 2 - radius, 0)
        y_max = min(image.shape[0] // 2 + radius, image.shape[0])
        y_min = max(image.shape[0] // 2 - radius, 0)
        image = image[y_min:y_max, x_min:x_max, :]
    return image
