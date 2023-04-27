"""
Test person registration route
"""
import pytest
from app.models import ModelType
from tests.conftest import MYSQL_TEST_TABLE, TEST_PERSON_FILE_ID, TEST_PERSON_URL_ID


@pytest.mark.asyncio
@pytest.mark.order(before=["test_person_recognition_route.py::test_recognition_one_person_file"])
async def test_registration_one_person_file(
        test_app_asyncio, test_mysql_connec, test_redis_connec, mock_one_face_image_1_file, mock_person_data_dict):
    """
    Test one person registration
    """
    # purge MYSQL_TEST_TABLE related cache first
    for key in test_redis_connec.keys(f"{MYSQL_TEST_TABLE}_*"):
        test_redis_connec.delete(key)
    fpath, fcontent = mock_one_face_image_1_file
    param_dict = mock_person_data_dict(TEST_PERSON_FILE_ID)

    files = [('img_file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post(
        "/register_person_file",
        files=files,
        params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_FILE_ID} registered into database"}


@pytest.mark.asyncio
@pytest.mark.order(before=["test_person_recognition_route.py::test_recognition_one_person_url"])
async def test_registration_one_person_url(
        test_app_asyncio, test_mysql_connec, test_redis_connec, mock_one_face_image_2_url, mock_person_data_dict):
    """
    Test one person registration
    """
    # purge MYSQL_TEST_TABLE related cache first
    for key in test_redis_connec.keys(f"{MYSQL_TEST_TABLE}_*"):
        test_redis_connec.delete(key)
    furl = mock_one_face_image_2_url
    param_dict = mock_person_data_dict(TEST_PERSON_URL_ID)

    param_dict["img_url"] = furl
    param_dict["model_type"]= ModelType.SLOW

    response = await test_app_asyncio.post(
        "/register_person_url",
        params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_URL_ID} registered into database"}
