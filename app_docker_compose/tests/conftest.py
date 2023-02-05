"""
Test configurations
"""
import sys
sys.path.append("app")

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.server import app


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
