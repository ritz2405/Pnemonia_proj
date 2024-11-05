import sys
sys.path.append('D:\\Full\\Pnemonia_proj')
from XRay.exception import XRayException
from XRay.pipeline.training_pipeline import TrainPipeline

def start_training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipline()
    except Exception as e:
        raise XRayException(e, sys)

if __name__ == "__main__":
    start_training()