import pytest


@pytest.mark.asyncio
async def test_server_root(test_app_asyncio):
    """
    Test if fastapi+uvicorn server is up
    """
    response = await test_app_asyncio.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Welcome to Person Face Registration & Recognition Service": "Please visit /docs for list of apis"}
