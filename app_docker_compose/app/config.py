"""
configurations and env variables load
"""
import os

ROOT_DOWNLOAD_PATH = os.getenv('ROOT_DOWNLOAD_PATH', default="app/data")

# redis conf
REDIS_HOST = os.getenv("REDIS_HOST", default="127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", default=6379)

# mysql conf
MYSQL_HOST = os.getenv("MYSQL_HOST", default="127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", default=3306)
MYSQL_USER = os.getenv("MYSQL_USER", default="user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", default="pass")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", default="default")
MYSQL_PERSON_TABLE = os.getenv("MYSQL_PERSON_TABLE", default="person")

# milvus conf
MILVUS_HOST = os.getenv("MILVUS_HOST", default="127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", default=19530)
FACE_VECTOR_DIM = 128
FACE_METRIC_TYPE = "L2"
FACE_INDEX_TYPE = "IVF_FLAT"
FACE_COLLECTION_NAME = 'faces'
