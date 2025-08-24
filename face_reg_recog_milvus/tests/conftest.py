"""
Test configurations
"""

import os
import sys
from datetime import date

import pymysql
import pytest
import pytest_asyncio
import redis
from httpx import ASGITransport, AsyncClient
from pymilvus import connections, utility
from pymysql.cursors import DictCursor

sys.path.append("app")

# custom settings
TEST_PERSON_FILE_ID = -1
TEST_PERSON_URL_ID = -2
TEST_PERSON_MYSQL_ID = -3
TEST_COLLECTION_NAME = "test"
MYSQL_TEST_TABLE = "test"
os.environ["MYSQL_CUR_TABLE"] = MYSQL_TEST_TABLE  # chg cur table for test duration

# custom imports
from app.api.milvus import get_milvus_collec_conn
from app.config import (
    FACE_INDEX_NLIST,
    FACE_INDEX_TYPE,
    FACE_METRIC_TYPE,
    FACE_VECTOR_DIM,
    MILVUS_HOST,
    MILVUS_PORT,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PERSON_TABLE,
    MYSQL_PORT,
    MYSQL_USER,
    REDIS_HOST,
    REDIS_PORT,
)
from app.server import app  # must be import after changing MYSQL_CUR_TABLE env var


def _load_file_content(fpath: str) -> bytes:
    """
    Load file from fpath and return as bytes
    """
    with open(fpath, "rb") as fptr:
        file_content = fptr.read()
    return file_content


@pytest_asyncio.fixture(scope="function")
async def test_app_asyncio():
    """
    Sets up the async server
    for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as aclient:
        yield aclient


@pytest.fixture(scope="session")
def test_milvus_connec():
    """Yields a milvus collection connection instance"""
    print("Setting milvus connection & creating collection if it already doesn't exist")
    milvus_collec_conn = get_milvus_collec_conn(
        collection_name=TEST_COLLECTION_NAME,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        vector_dim=FACE_VECTOR_DIM,
        metric_type=FACE_METRIC_TYPE,
        index_type=FACE_INDEX_TYPE,
        index_metric_params={"nlist": FACE_INDEX_NLIST},
    )
    milvus_collec_conn.load()
    yield milvus_collec_conn
    # drop test collections in teardown
    print("Tearing milvus connection")
    utility.drop_collection(TEST_COLLECTION_NAME)
    connections.disconnect("default")


@pytest.fixture(scope="session")
def test_mysql_connec():
    """Yields a mysql connection instance"""
    print("Setting mysql connection")
    mysql_conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DATABASE,
        cursorclass=DictCursor,
    )
    # create test table if not present & purge all existing data
    with mysql_conn.cursor() as cursor:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {MYSQL_TEST_TABLE} LIKE {MYSQL_PERSON_TABLE};")
        cursor.execute(f"DELETE FROM {MYSQL_TEST_TABLE}")
    mysql_conn.commit()
    yield mysql_conn
    # drop table in teardown
    print("Tearing mysql connection")
    with mysql_conn.cursor() as cursor:
        cursor.execute(f"DROP TABLE {MYSQL_TEST_TABLE}")
    mysql_conn.commit()
    mysql_conn.close()


@pytest.fixture(scope="session")
def test_redis_connec():
    """Yields a redis connection instance"""
    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    yield redis_conn
    # purge MYSQL_TEST_TABLE related cache in teardown
    for key in redis_conn.keys(f"{MYSQL_TEST_TABLE}_*"):
        redis_conn.delete(key)


@pytest.fixture(scope="session")
def mock_person_data_dict():
    """
    returns a func to create a person_data dict for testing
    """

    def _gen_data(person_id: int = -1):
        person_data = {
            "ID": person_id,
            "name": "bar",
            "birthdate": date(1971, 1, 30),
            "country": "foo",
            "city": "foobar",
            "title": "barfoo",
            "org": "foofoobar",
        }
        return person_data

    return _gen_data


@pytest.fixture(scope="session")
def mock_one_face_image_1_file():
    """
    load and return an image with a single face
    """
    fpath = "app/static/faces/one_face_1.jpg"
    return fpath, _load_file_content(fpath)


@pytest.fixture(scope="session")
def mock_one_face_image_2_file():
    """
    load and return an image with a single face
    """
    fpath = "app/static/faces/one_face_2.jpg"
    return fpath, _load_file_content(fpath)


@pytest.fixture(scope="session")
def mock_one_face_image_1_url():
    """
    returns an image url with a single face
    """
    return "https://raw.githubusercontent.com/SamSamhuns/face_registration_and_recognition_milvus/master/app_docker_compose/app/static/faces/one_face_1.jpg"


@pytest.fixture(scope="session")
def mock_one_face_image_2_url():
    """
    returns an image url with a single face
    """
    return "https://raw.githubusercontent.com/SamSamhuns/face_registration_and_recognition_milvus/master/app_docker_compose/app/static/faces/one_face_2.jpg"


@pytest.fixture(scope="session")
def mock_two_face_image():
    """
    load and return an image with a single face
    """
    fpath = "app/static/faces/two_faces.jpg"
    return fpath, _load_file_content(fpath)


@pytest.fixture(scope="session")
def mock_no_face_image():
    """
    load and return an image with a single face
    """
    fpath = "app/static/faces/no_face.jpg"
    return fpath, _load_file_content(fpath)
