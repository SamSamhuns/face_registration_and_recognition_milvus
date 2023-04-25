"""
Test configurations
"""
import sys
sys.path.append("app")

import pytest
import pytest_asyncio
from httpx import AsyncClient
from datetime import date

import redis
import pymysql
from pymysql.cursors import DictCursor
from pymilvus import connections

from app.server import app
from app.api.milvus import get_milvus_connec
from app.config import (
    REDIS_HOST, REDIS_PORT,
    MYSQL_HOST, MYSQL_PORT,
    MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    MILVUS_HOST, MILVUS_PORT, FACE_COLLECTION_NAME,
    FACE_VECTOR_DIM, FACE_METRIC_TYPE, FACE_INDEX_TYPE, FACE_COLLECTION_NAME)


def _load_file_content(fpath: str) -> bytes:
    """
    Load file from fpath and return as bytes
    """
    with open(fpath, 'rb') as fptr:
        file_content = fptr.read()
    return file_content


@pytest_asyncio.fixture(scope="function")
async def test_app_asyncio():
    # for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac  # testing happens here


@pytest.fixture
def test_milvus_connec():
    """Yields a milvus connection instance"""
    milvus_conn = get_milvus_connec(
        collection_name=FACE_COLLECTION_NAME,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        vector_dim=FACE_VECTOR_DIM,
        metric_type=FACE_METRIC_TYPE,
        index_type=FACE_INDEX_TYPE)
    milvus_conn.load()
    print("Setting milvus connection")
    yield milvus_conn
    print("Tearing milvus connection")
    connections.disconnect("default")


@pytest.fixture
def test_mysql_connec():
    """Yields a mysql connection instance"""
    mysql_conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DATABASE,
        cursorclass=DictCursor
    )
    print("Setting mysql connection")
    yield mysql_conn
    print("Tearing mysql connection")
    mysql_conn.close()


@pytest.fixture
def test_redis_connec():
    """Yields a redis connection instance"""
    redis_conn = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        decode_responses=True)
    yield redis_conn


@pytest.fixture(scope="session")
def mock_person_data_dict(person_id: int = 0):
    """
    create a person_data dict for testing
    """
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


@pytest.fixture(scope="session")
def mock_one_face_image():
    """
    load and return an image with a single face
    """
    fpath = "app/static/faces/one_face.jpg"
    return fpath, _load_file_content(fpath)


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
