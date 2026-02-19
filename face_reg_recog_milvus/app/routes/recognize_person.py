"""
Face Recognition fastapi file
"""

import logging
import os
import traceback
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.config import DOWNLOAD_CACHE_PATH, FACE_FEAT_MODEL_TYPE
from app.inference import recognize_person
from app.models import InputModel
from app.utils import cache_file_locally, download_url_file, get_mode_ext, remove_file

router = APIRouter()
logger = logging.getLogger("recognize_person_route")


class RecognizePersonProcessTask:
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
            face_dist_threshold=self.input_data.face_dist_threshold,
        )
        self.response_data = {**results}


@router.post("/recognize_person_file")
async def recognize_person_file(background_tasks: BackgroundTasks, img_file: UploadFile = File(...)):
    """
    recognize person from the face image file uploaded as a file
    """
    response_data = {}
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = await img_file.read()
        file_cache_path = os.path.join(DOWNLOAD_CACHE_PATH, file_name)

        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(model_name=FACE_FEAT_MODEL_TYPE.name, file_path=file_cache_path)
        task = RecognizePersonProcessTask(recognize_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        logger.error("%s: %s", excep, traceback.print_exc())
        response_data["message"] = "failed to recognize face from image"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep

    return response_data


@router.post("/recognize_person_url")
async def recognize_person_url(background_tasks: BackgroundTasks, img_url: str):
    """
    recognize person from the face image file provided as a url
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
        input_data = InputModel(model_name=FACE_FEAT_MODEL_TYPE.name, file_path=file_cache_path)
        task = RecognizePersonProcessTask(recognize_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        logger.error("%s: %s", excep, traceback.print_exc())
        response_data["message"] = f"failed to recognize face  from image downloaded from {img_url}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep

    return response_data
