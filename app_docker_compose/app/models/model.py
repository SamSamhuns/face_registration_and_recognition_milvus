"""
data models for fastapi+uvicorn server
"""
from datetime import date
from enum import Enum
from pydantic import BaseModel, ConfigDict
from collections import namedtuple


Model = namedtuple('Model', ['name', 'dim'])


class PersonModel(BaseModel):
    """
    Person data model. Based on the person table schema
    id: int = must be a unique id in the database, required
    name: str = name of person, required
    birthdate: str = date with format YYYY-MM-DD, required
    country: str = country, required
    city: str = city, optional
    title: str = person's title, optional
    org: str = person's org, optional
    """
    ID: int
    name: str
    birthdate: date
    country: str
    city: str = ""
    title: str = ""
    org: str = ""


class InputModel(BaseModel):
    """
    API input model format
    """
    model_name: str
    file_path: str
    face_det_threshold: float = 0.3
    face_dist_threshold: float = 10
    person_data: PersonModel = None

    model_config = ConfigDict(
        protected_namespaces=('restricted_')
    )


class ModelType(Model, Enum):
    """
    Face feature model name and vector dimension
    """
    FACE_REID = Model("face_reid_retail_0095", 256)
    FACENET = Model("facenet", 128)
    ARCFACE = Model("arcface_resnet18_110", 512)
