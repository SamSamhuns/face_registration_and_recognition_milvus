import os
import uuid
import traceback

from fastapi import APIRouter
from fastapi import UploadFile, File, Form, BackgroundTasks

from inference import register_face
from models import InputModel, ModelType
from utils import get_mode_ext, remove_file, download_url_file, cache_file_locally


router = APIRouter()
TEMP_DOWNLOAD_URL = os.getenv('TEMP_DOWNLOAD_URL')


class RegisterFaceProcessTask():
    __slots__ = ["func", "input_data", "response_data"]

    def __init__(self, func, input_data):
        super(RegisterFaceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

    def run(self):
        # run func and get results as dict
        results = self.func(
            input_file=self.input_data.file_path,
            model_name=self.input_data.model_name,
            inference_mode=self.input_data.inference_mode,
            threshold=self.input_data.threshold)
        self.response_data = {**results}


@router.post("/register_face_file")
async def register_face_file(background_tasks: BackgroundTasks,
                             model_type: ModelType,
                             person_name: str,
                             file: UploadFile = File(...)):
    response_data = dict()
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = file.file.read()
        file_cache_path = os.path.join(TEMP_DOWNLOAD_URL, file_name)
        os.makedirs(TEMP_DOWNLOAD_URL, exist_ok=True)
        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(model_type.value, "image", file_cache_path)
        task = RegisterFaceProcessTask(register_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to register uploaded image to server"

    return response_data


@router.post("/register_face_url")
async def register_face_url(background_tasks: BackgroundTasks,
                            model_type: ModelType,
                            person_name: str,
                            url: str):
    response_data = dict()
    try:
        os.makedirs(TEMP_DOWNLOAD_URL, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_cache_path = os.path.join(TEMP_DOWNLOAD_URL, file_name)
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download image from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(model_type.value, "image", file_cache_path)
        task = RegisterFaceProcessTask(register_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to register url image from {url} to server"

    return response_data
