import os
import sys
sys.path.append('D:\\Full\\Pnemonia_proj')
from XRay.exception import XRayException

def get_data_from_s3(self):
    try:
        pass
    except Exception as e:
        raise XRayException(e, sys) 