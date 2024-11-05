from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass
class DataIngestionArtifact:
    train_file_path: str

    test_file_path: str
    val_file_path: str
@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    transformed_val_object: DataLoader

    train_transforms_path: str
    test_transforms_path: str
    val_transforms_path: str