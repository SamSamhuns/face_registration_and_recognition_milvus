import os
import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import face, recognize_face, register_face


# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = os.path.join(ROOT, ".data_cache")


def get_application(title="Face Registration and Recognition"):
    app = FastAPI(title=title, version="1.0.0")
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


@app.get("/")
async def index():
    return {"Welcome to Face Registration & Recognition Service": "Please visit /docs for list of apis"}


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
