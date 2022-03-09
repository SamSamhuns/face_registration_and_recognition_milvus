import os
import uuid
import argparse
import traceback
from enum import Enum
import urllib.request as urllib2

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks

from inference import run_inference


# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = os.path.join(ROOT, ".data_cache")

app = FastAPI(title="Face Feature Extraction and Recognition")

# load models here
model1 = "facenet"


class InputModel(BaseModel):
    model_name: str
    inference_mode: str
    file_path: str
    threshold: float


class ModelName(str, Enum):
    facenet = model1


class InferenceMode(str, Enum):
    image = "image"
    video = "video"


def get_mode_ext(mode):
    return {"image": ".jpg", "video": ".mp4"}[mode]


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def download_url_file(download_url: str, download_path: str) -> None:
    response = urllib2.urlopen(download_url)
    with open(download_path, 'wb') as f:
        f.write(response.read())


async def cache_file_locally(file_cache_path: str, data: bytes) -> None:
    with open(file_cache_path, 'wb') as img_file_ptr:
        img_file_ptr.write(data)


class InferenceProcessTask():
    def __init__(self, func, input_data):
        super(InferenceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

    def run(self):
        # run the inference function
        self.results = self.func(
            input_file=self.input_data.file_path,
            model_name=self.input_data.model_name,
            threshold=self.input_data.threshold,
            inference_mode=self.input_data.inference_mode)
        self.response_data["code"] = "success"
        self.response_data['msg'] = "prediction successful"
        self.response_data["prediction"] = self.results


@app.post("/inference_file")
async def inference_file(input_model: ModelName,
                         inference_mode: InferenceMode,
                         background_tasks: BackgroundTasks,
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
            model_name=input_model.value,
            inference_mode=inference_mode.value,
            file_path=file_cache_path,
            threshold=threshold)
        task = InferenceProcessTask(
            run_inference,
            input_data=input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to run inference on uploaded {inference_mode.value}"

    return response_data


@app.post("/inference_url")
async def inference_url(input_model: ModelName,
                        inference_mode: InferenceMode,
                        background_tasks: BackgroundTasks,
                        url: str = Form(""),
                        threshold: float = Form(0.30)):
    response_data = dict()
    os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
    file_name = str(uuid.uuid4()) + get_mode_ext(inference_mode.value)
    file_cache_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
    os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)

    try:
        download_url_file(url, file_cache_path)
        background_tasks.add_task(remove_file, file_cache_path)
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data['msg'] = f"couldn't download {inference_mode.value} from \'{url}\'. Not a valid link."
        return response_data

    try:
        input_data = InputModel(
            model_name=input_model.value,
            inference_mode=inference_mode.value,
            file_path=file_cache_path,
            threshold=threshold)
        task = InferenceProcessTask(
            run_inference,
            input_data=input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e, traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = f"failed to run inference on {inference_mode} from {url}"

    return response_data


@app.get("/")
def index():
    return {"Welcome to Face Recognition Service": "Please visit /docs"}


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
