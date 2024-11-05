from datetime import datetime
from typing import List
import torch

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

#Data Ingestion Constants

ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "xraylungsimgs"
S3_DATA_FOLDER : str = "uploadable"

#Data Transformation

CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"
BRIGHTNESS: float = 0.10
CONTRAST: float = 0.1

SATURATION: float = 0.10

HUE: float = 0.1

RESIZE: int = 224

CENTERCROP: int = 224

RANDOMROTATION: int = 10

NORMALIZE_LIST_1: List[float] = [0.485, 0.456, 0.406]

NORMALIZE_LIST_2: List[float] = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"

TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"

TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

VAL_TRANSFORMS_FILE: str = "val_transforms.pkl"

BATCH_SIZE: int = 2

SHUFFLE: bool = False

PIN_MEMORY: bool = True

