import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from data_handling.utils import _btgraham_processing
from default_paths import (
    DATA_DIR_RSNA,
    DATA_DIR_RSNA_PROCESSED_IMAGES,
    DATA_DIR_DIABETIC,
    DATA_DIR_DIABETIC_PROCESSED_IMAGES,
)
from PIL import Image
from torchvision.transforms import Resize


def run_rsna_preprocessing_script():
    tf = Resize(224)
    root = DATA_DIR_RSNA

    df = pd.read_csv(
        root / "stage_2_train_images" / "stage_2_train_labels.csv"
    ).drop_duplicates()
    DATA_DIR_RSNA_PROCESSED_IMAGES.mkdir(parents=True, exist_ok=True)
    subject_ids = df.patientId.values
    filenames = [root / f"{subject_id}.dcm" for subject_id in subject_ids]
    for file in filenames:
        scan_image = pydicom.filereader.dcmread(file).pixel_array.astype(np.float32)
        scan_image = (
            (scan_image - scan_image.min())
            * 255.0
            / (scan_image.max() - scan_image.min())
        )
        image = Image.fromarray(scan_image).convert("L")
        image = tf(image)
        image.save(DATA_DIR_RSNA_PROCESSED_IMAGES / str(file.stem + ".png"))


def run_diabetic_preprocessing_script():
    (DATA_DIR_DIABETIC_PROCESSED_IMAGES / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR_DIABETIC_PROCESSED_IMAGES / "test").mkdir(parents=True, exist_ok=True)
    train_files = [
        DATA_DIR_DIABETIC / "train" / f"{file}.jpeg"
        for file in pd.read_csv(DATA_DIR_DIABETIC / "trainLabels.csv").image.values
    ]
    for file in tqdm(train_files):
        _btgraham_processing(
            filepath=file, extract_dir=DATA_DIR_DIABETIC_PROCESSED_IMAGES / "train"
        )

    val_test_files = [
        DATA_DIR_DIABETIC / "test" / f"{file}.jpeg"
        for file in pd.read_csv(
            DATA_DIR_DIABETIC / "retinopathy_solution.csv"
        ).image.values
    ]
    for file in tqdm(val_test_files):
        _btgraham_processing(
            filepath=file, extract_dir=DATA_DIR_DIABETIC_PROCESSED_IMAGES / "test"
        )


if __name__ == "__main__":
    run_diabetic_preprocessing_script()
