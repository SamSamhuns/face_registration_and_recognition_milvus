from starlette.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_default():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome to Face Registration & Recognition Service": "Please visit /docs for list of apis"}
