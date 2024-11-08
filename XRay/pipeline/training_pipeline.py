import sys
sys.path.append('D:\\Full\\Pnemonia_proj')
from XRay.components.data_ingestion import DataIngestion
from XRay.components.data_transformation import DataTransformation
from XRay.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from XRay.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
# from XRay.components.data_transformation import DataTransformation
# from XRay.components.model_training import ModelTrainer
# from XRay.components.model_evaluation import ModelEvaluation
# from XRay.components.model_pusher import ModelPusher
from XRay.exception import XRayException
from XRay.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    def start_data_ingestion(self)-> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from s3 bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from s3")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise XRayException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact)->DataTransformationArtifact:
        try: 
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)

    def run_pipline(self)->None:
        logging.info("Enetered the run_pipeline method of train Pipeline class")
        try:
            data_ingestion_artifact : DataIngestionArtifact = self.start_data_ingestion()
            data_transformation_artifact: DataTransformationArtifact = (
                self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )
            logging.info("Exited the run_pipeline method")
        except Exception as e:
            raise XRayException(e, sys)