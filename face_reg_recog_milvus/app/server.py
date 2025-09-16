"""
Main fastapi server file
"""

import argparse
import logging
import os
import time

import uvicorn
from config import FASTAPI_SERVER_PORT
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes import person, recognize_person, register_person

# logging
logger = logging.getLogger("server")


def get_application(title="Face Registration and Recognition"):
    """Gets FastAPI application object with CORS enabled."""
    fastapi_app = FastAPI(title=title, version="1.0.0")
    fastapi_app.mount("/static", StaticFiles(directory="./app/static"), name="static")
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fastapi_app


app = get_application()
app.include_router(person.router)
app.include_router(recognize_person.router)
app.include_router(register_person.router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds an X-Process-Time header to the response indicating the api request processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def index():
    """Root endpoint. Returns a welcome message."""
    return {"Welcome to Person Face Registration & Recognition Service": "Please visit /docs for list of apis"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "healthy"}


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint. Returns the favicon."""
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "app/static", file_name)
    return FileResponse(path=file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Start FastAPI with uvicorn server hosting inference models""")
    parser.add_argument("-ip", "--host_ip", type=str, default="0.0.0.0", help="host ip address. (default: %(default)s)")
    parser.add_argument(
        "-p", "--port", type=int, default=FASTAPI_SERVER_PORT, help="uvicorn port number. (default: %(default)s)"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="number of uvicorn workers. (default: %(default)s)"
    )
    args = parser.parse_args()

    logger.info("Uvicorn server running on %s:%s with %s workers", args.host_ip, args.port, args.workers)
    uvicorn.run("server:app", host=args.host_ip, port=args.port, workers=args.workers, reload=True, reload_dirs=["app"])
