"""
Face Registration fastapi file
"""

import logging
import os
import traceback
import uuid

from config import DOWNLOAD_CACHE_PATH, FACE_FEAT_MODEL_TYPE
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.inference import register_person
from app.models import InputModel, PersonModel
from app.utils import cache_file_locally, download_url_file, get_mode_ext, remove_file

router = APIRouter()
logger = logging.getLogger("register_person_router")


class RegisterPersonProcessTask:
    """
    Stores input data and api funcs
    """

    __slots__ = ["func", "input_data", "response_data"]

    def __init__(self, func, input_data):
        super().__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = {}

    def run(self):
        """run func and get results as dict"""
        results = self.func(
            model_name=self.input_data.model_name,
            file_path=self.input_data.file_path,
            face_det_threshold=self.input_data.face_det_threshold,
            person_data=self.input_data.person_data.model_dump(),
        )
        self.response_data = {**results}


@router.post("/register_person_file")
async def register_person_file(
    background_tasks: BackgroundTasks, person_data: PersonModel = Depends(), img_file: UploadFile = File(...)
):
    """
    registers person face and info with the face uploaded as an image file

    Person data
        id: int = must be a unique id in the database, required
        name: str = name of person, required
        birthdate: str = date with format YYYY-MM-DD, required
        country: str = country, required
        city: str = city, optional
        title: str = person's title, optional
        org: str = person's org, optional
    """
    response_data = {}
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = img_file.file.read()
        file_cache_path = os.path.join(DOWNLOAD_CACHE_PATH, file_name)

        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(
            model_name=FACE_FEAT_MODEL_TYPE.name, file_path=file_cache_path, person_data=person_data
        )
        task = RegisterPersonProcessTask(register_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        logger.error("%s: %s", excep, traceback.print_exc())
        response_data["message"] = "failed to register uploaded image to server"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep

    return response_data


@router.post("/register_person_url")
async def register_person_url(background_tasks: BackgroundTasks, img_url: str, person_data: PersonModel = Depends()):
    """
    registers person face and info with the face image file provided as a url

    Person data
        id: int = must be a unique id in the database, required
        name: str = name of person, required
        birthdate: str = date with format YYYY-MM-DD, required
        country: str = country, required
        city: str = city, optional
        title: str = person's title, optional
        org: str = person's org, optional
    """
    response_data = {}
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_cache_path = os.path.join(DOWNLOAD_CACHE_PATH, file_name)

        await download_url_file(img_url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as excep:
        logger.error("%s: %s", excep, traceback.print_exc())
        response_data["message"] = f"couldn't download image from '{img_url}'. Not a valid link."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep

    try:
        input_data = InputModel(
            model_name=FACE_FEAT_MODEL_TYPE.name, file_path=file_cache_path, person_data=person_data
        )
        task = RegisterPersonProcessTask(register_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        logger.error("%s: %s", excep, traceback.print_exc())
        response_data["message"] = f"failed to register url image from {img_url} to server"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep

    return response_data
