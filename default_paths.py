from pathlib import Path

DATA_DIR_RSNA = Path("/vol/biodata/data/chest_xray/rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"

DATA_DIR_DIABETIC = Path("/vol/biodata/data/diabetic_retino")
DATA_DIR_DIABETIC_PROCESSED_IMAGES = DATA_DIR_DIABETIC / "preprocess_as_tensorflow"


DATA_MEDMNIST = Path(__file__).parent / "data"
DATA_BUSI = Path("/data/failure_detection/data/BUSI/Dataset_BUSI_with_GT")
