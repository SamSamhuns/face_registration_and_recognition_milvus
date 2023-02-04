import os
import uuid
import traceback

from fastapi import APIRouter
from fastapi import UploadFile, File, Form, BackgroundTasks

from models import InputModel, ModelType
from utils import get_mode_ext, remove_file, download_url_file, cache_file_locally
from inference import recognize_face


router = APIRouter()
ROOT_DOWNLOAD_URL = os.getenv('ROOT_DOWNLOAD_URL')


class RecognizeFaceProcessTask():
    __slots__ = ["func", "input_data", "response_data"]

    def __init__(self, func, input_data):
        super(RecognizeFaceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

    def run(self):
        # run func and get results as dict
        results = self.func(
            model_name=self.input_data.model_name,
            file_path=self.input_data.file_path,
            threshold=self.input_data.threshold)
        self.response_data = {**results}


@router.post("/recognize_face_file")
async def recognize_face_file(background_tasks: BackgroundTasks,
                              file: UploadFile = File(...)):
    response_data = dict()
    model_type: ModelType = ModelType.SLOW  # default to SLOW for now
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = file.file.read()
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)

        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(model_name=model_type.value, file_path=file_cache_path, person_name="")
        task = RecognizeFaceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to recognize face from image"

    return response_data


@router.post("/recognize_face_url")
async def recognize_face_url(background_tasks: BackgroundTasks,
                             url: str):
    response_data = dict()
    model_type: ModelType = ModelType.SLOW  # default to SLOW for now
    try:
        os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download image from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(model_name=model_type.value, file_path=file_cache_path)
        task = RecognizeFaceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to recognize face  from image downloaded from {url}"

    return response_data