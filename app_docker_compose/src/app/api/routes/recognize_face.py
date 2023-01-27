import os
import uuid
import traceback

from fastapi import APIRouter
from fastapi import UploadFile, File, Form, BackgroundTasks

from models import InputModel, InferenceMode, ModelType
from utils import get_mode_ext, remove_file, download_url_file, cache_file_locally
from inference import recognize_face


router = APIRouter()
TEMP_DOWNLOAD_URL = os.getenv('TEMP_DOWNLOAD_URL')


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
            input_file=self.input_data.file_path,
            model_name=self.input_data.model_name,
            inference_mode=self.input_data.inference_mode,
            threshold=self.input_data.threshold)
        self.response_data = {**results}


@router.post("/recognize_face_file")
async def recognize_face_file(background_tasks: BackgroundTasks,
                              model_type: ModelType,
                              inference_mode: InferenceMode,
                              file: UploadFile = File(...),
                              threshold: float = Form(0.30)):
    response_data = dict()
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext(inference_mode.value)
        file_bytes_content = file.file.read()
        file_cache_path = os.path.join(TEMP_DOWNLOAD_URL, file_name)
        os.makedirs(TEMP_DOWNLOAD_URL, exist_ok=True)
        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(
            model_type.value, inference_mode.value, file_cache_path, threshold)
        task = RecognizeFaceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to recognize face from {inference_mode.value}"

    return response_data


@router.post("/recognize_face_url")
async def recognize_face_url(background_tasks: BackgroundTasks,
                             model_type: ModelType,
                             inference_mode: InferenceMode,
                             url: str,
                             threshold: float = Form(0.30)):
    response_data = dict()
    try:
        os.makedirs(TEMP_DOWNLOAD_URL, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext(inference_mode.value)
        file_cache_path = os.path.join(TEMP_DOWNLOAD_URL, file_name)
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download {inference_mode.value} from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(
            model_type.value, inference_mode.value, file_cache_path, threshold)
        task = RecognizeFaceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as excep:
        print(excep, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to recognize face  from {inference_mode} downloaded from {url}"

    return response_data
