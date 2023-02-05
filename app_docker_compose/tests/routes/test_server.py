import pytest


@pytest.mark.asyncio
async def test_server_root(test_app_asyncio):
    response = await test_app_asyncio.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Welcome to Face Registration & Recognition Service": "Please visit /docs for list of apis"}
