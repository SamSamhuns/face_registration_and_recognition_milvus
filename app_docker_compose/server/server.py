import os
import uuid
import argparse
import traceback
from enum import Enum

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks

from inference import register_face, recognize_face
from utils import get_mode_ext, remove_file, download_url_file, cache_file_locally


# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = os.path.join(ROOT, ".data_cache")

app = FastAPI(title="Face Registration and Recognition")


class InputModel(BaseModel):
    model_name: str
    inference_mode: str
    file_path: str
    threshold: float = 0.3


class ModelType(str, Enum):
    cpu = "CPU"
    gpu = "GPU"


class InferenceMode(str, Enum):
    image = "image"
    video = "video"


class InferenceProcessTask():
    __slots__ = ["func", "input_data", "response_data"]

    def __init__(self, func, input_data):
        super(InferenceProcessTask, self).__init__()
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


@app.post("/register_face_file")
async def register_face_file(background_tasks: BackgroundTasks,
                             model_type: ModelType,
                             person_name: str,
                             file: UploadFile = File(...)):
    response_data = dict()
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_bytes_content = file.file.read()
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(model_type.value, "image", file_cache_path)
        task = InferenceProcessTask(register_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to register uploaded image to server"

    return response_data


@app.post("/register_face_url")
async def register_face_url(background_tasks: BackgroundTasks,
                            model_type: ModelType,
                            person_name: str,
                            url: str = Form("")):
    response_data = dict()
    try:
        os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext("image")
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download image from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(model_type.value, "image", file_cache_path)
        task = InferenceProcessTask(register_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to register url image from {url} to server"

    return response_data


@app.post("/recognize_face_file/{inference_mode}")
async def recognize_face_file(background_tasks: BackgroundTasks,
                              inference_mode: InferenceMode,
                              model_type: ModelType,
                              file: UploadFile = File(...),
                              threshold: float = Form(0.30)):
    response_data = dict()
    try:
        file_name = str(uuid.uuid4()) + get_mode_ext(inference_mode.value)
        file_bytes_content = file.file.read()
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
        await cache_file_locally(file_cache_path, file_bytes_content)
        background_tasks.add_task(remove_file, file_cache_path)

        input_data = InputModel(
            model_type.value, inference_mode.value, file_cache_path, threshold)
        task = InferenceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to recognize face from {inference_mode.value}"

    return response_data


@app.post("/recognize_face_url/{inference_mode}")
async def recognize_face_url(background_tasks: BackgroundTasks,
                             inference_mode: InferenceMode,
                             model_type: ModelType,
                             url: str = Form(""),
                             threshold: float = Form(0.30)):
    response_data = dict()
    try:
        os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
        file_name = str(uuid.uuid4()) + get_mode_ext(inference_mode.value)
        file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download {inference_mode.value} from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(
            model_type.value, inference_mode.value, file_cache_path, threshold)
        task = InferenceProcessTask(recognize_face, input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to recognize face  from {inference_mode} downloaded from {url}"

    return response_data


@app.get("/")
def registered_faces():
    return {"TODO: Should return all registered faces and corresponding assigned name"}


@app.get("/")
def index():
    return {"Welcome to Face Recognition Service": "Please visit /docs for list of apis"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Start FastAPI with uvicorn server hosting inference models""")
    parser.add_argument('-ip', '--host_ip', type=str, default="0.0.0.0",
                        help='host ip address. (default: %(default)s)')
    parser.add_argument('-p', '--port', type=int, default=8080,
                        help='uvicorn port number. (default: %(default)s)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="number of uvicorn workers. (default: %(default)s)")
    args = parser.parse_args()

    print(
        f"Uvicorn server running on {args.host_ip}:{args.port} with {args.workers} workers")
    uvicorn.run(app, host=args.host_ip, port=args.port, workers=args.workers)
