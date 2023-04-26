"""
Face Registration fastapi file
"""
import os
import uuid
import traceback

from fastapi import APIRouter
from fastapi import UploadFile, File, Depends, BackgroundTasks

from inference import register_person
from models import InputModel, PersonModel, ModelType
from utils import get_mode_ext, remove_file, download_url_file, cache_file_locally
from config import ROOT_DOWNLOAD_PATH


router = APIRouter()


class RegisterPersonProcessTask():
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
            threshold=self.input_data.threshold,
            person_data=self.input_data.person_data.dict())
        self.response_data = {**results}


@router.post("/register_person_file")
async def register_person_file(background_tasks: BackgroundTasks,
                               person_data: PersonModel = Depends(),
                               img_file: UploadFile = File(...)):
    """
    registers person face and info with the face uploaded as an image file
    """
    response_data = {}
    model_type: ModelType = ModelType.SLOW  # default to SLOW for now
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = img_file.file.read()
        file_cache_path = os.path.join(ROOT_DOWNLOAD_PATH, file_name)

        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(model_name=model_type.value,
                                file_path=file_cache_path,
                                person_data=person_data)
        task = RegisterPersonProcessTask(register_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["status"] = "failed"
        response_data["message"] = "failed to register uploaded image to server"

    return response_data


@router.post("/register_person_url")
async def register_person_url(background_tasks: BackgroundTasks,
                              model_type: ModelType,
                              img_url: str,
                              person_data: PersonModel = Depends()):
    """
    registers person face and info with the face image file provided as a url
    """
    response_data = {}
    try:
        os.makedirs(ROOT_DOWNLOAD_PATH, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_cache_path = os.path.join(ROOT_DOWNLOAD_PATH, file_name)

        await download_url_file(img_url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["status"] = "failed"
        response_data["message"] = f"couldn't download image from \'{img_url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(model_name=model_type.value,
                                file_path=file_cache_path,
                                person_data=person_data)
        task = RegisterPersonProcessTask(register_person, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["status"] = "failed"
        response_data["message"] = f"failed to register url image from {img_url} to server"

    return response_data
