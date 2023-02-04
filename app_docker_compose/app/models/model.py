from enum import Enum
from pydantic import BaseModel


class InputModel(BaseModel):
    """
    API input moel format
    """
    model_name: str
    file_path: str
    person_name: str
    threshold: float = 0.3


class ModelType(str, Enum):
    """
    Face feature extraction model type
    """
    FAST = "face-reidentification-retail-0095"
    SLOW = "facenet_trtserver"
