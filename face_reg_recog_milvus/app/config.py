"""
configurations and env variables load
"""

import os
from logging.config import dictConfig

from models import ModelType
from models.logging import LogConfig

# save directories
DOWNLOAD_CACHE_PATH = os.getenv("DOWNLOAD_CACHE_PATH", default="app/.data")
DOWNLOAD_IMAGE_PATH = os.getenv("DOWNLOAD_IMAGE_PATH", default="volumes/person_images")
LOG_STORAGE_PATH = os.getenv("LOG_STORAGE_PATH", default="volumes/server_logs")

os.makedirs(DOWNLOAD_CACHE_PATH, exist_ok=True)
os.makedirs(DOWNLOAD_IMAGE_PATH, exist_ok=True)
os.makedirs(LOG_STORAGE_PATH, exist_ok=True)

# logging conf
log_cfg = LogConfig()
# override info & error log paths
log_cfg.handlers["info_rotating_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "info.log")
log_cfg.handlers["warning_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "error.log")
log_cfg.handlers["error_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "error.log")
dictConfig(log_cfg.model_dump())

# http api server
API_SERVER_PORT = int(os.getenv("API_SERVER_PORT", default="8080"))

# redis conf
REDIS_HOST = os.getenv("REDIS_HOST", default="0.0.0.0")
REDIS_PORT = int(os.getenv("REDIS_PORT", default="6379"))

# mysql conf
MYSQL_HOST = os.getenv("MYSQL_HOST", default="0.0.0.0")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", default="3306"))
MYSQL_USER = os.getenv("MYSQL_USER", default="user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", default="pass")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", default="default")
MYSQL_PERSON_TABLE = os.getenv("MYSQL_PERSON_TABLE", default="person")
# table where ops will be run on
MYSQL_CUR_TABLE = os.getenv("MYSQL_CUR_TABLE", default=MYSQL_PERSON_TABLE)

# milvus conf
MILVUS_HOST = os.getenv("MILVUS_HOST", default="0.0.0.0")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", default="19530"))
FACE_FEAT_MODEL_TYPE = ModelType.ARCFACE
FACE_VECTOR_DIM = FACE_FEAT_MODEL_TYPE.dim
FACE_METRIC_TYPE = "L2"
FACE_INDEX_TYPE = "IVF_FLAT"
FACE_COLLECTION_NAME = "faces"
# num of clusters/buckets for each index specific to IVF_FLAT
FACE_INDEX_NLIST = 4096
# nprobe specific to IVF denotes num of closest buckets/clusters looked into per file
FACE_SEARCH_NPROBE = 2056
