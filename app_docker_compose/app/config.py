"""
configurations and env variables load
"""
import os

DOWNLOAD_CACHE_PATH = os.getenv('DOWNLOAD_CACHE_PATH', default="app/.data")
DOWNLOAD_IMAGE_PATH = os.getenv('DOWNLOAD_IMAGE_PATH', default="volumes/person_images")

# http api server
API_SERVER_PORT=int(os.getenv("API_SERVER_PORT", default="8080"))

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
FACE_VECTOR_DIM = 128
FACE_METRIC_TYPE = "L2"
FACE_INDEX_TYPE = "IVF_FLAT"
FACE_COLLECTION_NAME = 'faces'
# num of clusters/buckets for each index specific to IVF_FLAT
FACE_INDEX_NLIST = 4096
# nprobe specific to IVF denotes num of closest buckets/clusters looked into per file
FACE_SEARCH_NPROBE = 2056
