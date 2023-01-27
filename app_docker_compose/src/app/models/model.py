from enum import Enum
from pydantic import BaseModel


class InputModel(BaseModel):
    model_name: str
    inference_mode: str
    file_path: str
    threshold: float = 0.3


class ModelType(str, Enum):
    fast = "mobilenetV2SSD_with_faceReid"
    slow = "yoloV5Face_with_faceNet"


class InferenceMode(str, Enum):
    image = "image"
    video = "video"
