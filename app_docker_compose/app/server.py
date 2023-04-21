"""
Main fastapi server file
"""
import os
import time
import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.routes import face, recognize_face, register_face


# The root_path is the absolute path of the __init_.py under the source
root_path = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
root_download_path = os.path.join(root_path, "data")
os.environ["ROOT_DOWNLOAD_PATH"] = root_download_path
os.makedirs(root_download_path, exist_ok=True)


def get_application(title="Face Registration and Recognition"):
    app = FastAPI(title=title, version="1.0.0")
    app.mount("/static", StaticFiles(directory="./app/static"), name="static")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = get_application()
app.include_router(face.router)
app.include_router(recognize_face.router)
app.include_router(register_face.router)


# api call time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def index():
    return {"Welcome to Face Registration & Recognition Service": "Please visit /docs for list of apis"}


@app.get('/favicon.ico')
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "app/static", file_name)
    return FileResponse(path=file_path)


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
    uvicorn.run("server:app", host=args.host_ip, port=args.port,
                workers=args.workers, reload=True)
