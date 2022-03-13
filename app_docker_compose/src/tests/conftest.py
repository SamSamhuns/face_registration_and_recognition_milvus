import pytest
from starlette.testclient import TestClient

from app.server import app


@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client  # testing happens here


def test_default(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome to Face Registration & Recognition Service": "Please visit /docs for list of apis"}
