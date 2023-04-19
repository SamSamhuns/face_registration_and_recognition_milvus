from datetime import date
from enum import Enum
from pydantic import BaseModel


class PersonModel(BaseModel):
    """
    Person data model. Based on the person table schema
    """
    id: int
    name: str
    birthdate: date
    country: str
    city: str
    title: str
    org: str


class InputModel(BaseModel):
    """
    API input model format
    """
    model_name: str
    file_path: str
    threshold: float = 0.3
    person_data: PersonModel


class ModelType(str, Enum):
    """
    Face feature extraction model type
    """
    FAST = "face-reidentification-retail-0095"
    SLOW = "facenet_trtserver"
