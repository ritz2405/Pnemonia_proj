import os
import sys
sys.path.append('D:\\Full\\Pnemonia_proj')
from typing import Tuple
import joblib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from XRay.entity.artifact_entity import(
    DataIngestionArtifact,
    DataTransformationArtifact
)
from XRay.entity.config_entity import DataTransformationConfig
from XRay.exception import XRayException
from XRay.logger import logging

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
    
    def transforming_training_data(self)-> transforms.Compose:
        try:
            logging.info("Entered in transforming_training_data of DataTransformation class")
            train_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCorp(self.data_transformation_config.CENTERCORP),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )
            logging.info("Exited from transforming_training_data of DataTransformation class")
            return train_transform
        except Exception as e:
            raise XRayException(e, sys)
        
    def transforming_test_data(self)-> transforms.Compose:
        try:
            logging.info("Entered in transforming_test_data of DataTransformation class")
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCorp(self.data_transformation_config.CENTERCORP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )
            logging.info("Exited from transforming_test_data of DataTransformation class")
            return test_transform
        except Exception as e:
            raise XRayException(e, sys)
        
    def transforming_val_data(self)-> transforms.Compose:
        try:
            logging.info("Entered in transforming_val_data of DataTransformation class")
            val_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCorp(self.data_transformation_config.CENTERCORP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )
            logging.info("Exited from transforming_val_data of DataTransformation class")
            return val_transform
        except Exception as e:
            raise XRayException(e, sys)
    
    def data_loader(
            self, train_transform: transforms.Compose, test_transform: transforms.Compose, val_transform: transforms.Compose
    )-> Tuple[DataLoader, DataLoader, DataLoader]:
        try:
            logging.info("Entered in data_loader of DataTransformation class")
            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform
            )
            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform
            )
            val_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.val_file_path),
                transform=val_transform
            )
            logging.info("Created train, test and val paths")
            train_loader: DataLoader = DataLoader(
                train_data, **self.data_transformation_config.data_loader_params
            )
            test_loader: DataLoader = DataLoader(
                test_data, **self.data_transformation_config.data_loader_params
            )
            val_loadder: DataLoader = DataLoader(
                val_data, **self.data_transformation_config.data_loader_params,
            )
            logging.info("Exited the data_loader method of Data transformation class")
            return train_loader, test_loader, val_loadder
        except Exception as e:
            raise XRayException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            train_transform: transforms.Compose = self.transforming_training_data
            test_transform: transforms.Compose = self.transforming_test_data
            val_transform: transforms.Compose = self.transforming_val_data
            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)
            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )
            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )
            joblib.dump(
                val_transform, self.data_transformation_config.val_transforms_file
            )
            train_loader, test_loader, val_loader = self.data_loader(train_transform, test_transform, val_transform)
            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                transformed_val_object=val_loader,
                train_transforms_path=self.data_transformation_config.train_transforms_file,
                test_transforms_path=self.data_transformation_config.test_transforms_file,
                val_transforms_path=self.data_transformation_config.val_transforms_file
            )
            logging.info("Exited the initiate_data_transformation method of Data transformation class")
            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)
