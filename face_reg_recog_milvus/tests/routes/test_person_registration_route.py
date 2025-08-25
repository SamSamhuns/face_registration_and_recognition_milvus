"""
Test person registration route
"""
from unittest.mock import patch

import pytest
from tests.conftest import MYSQL_TEST_TABLE, TEST_PERSON_FILE_ID, TEST_PERSON_URL_ID


@pytest.mark.asyncio
@pytest.mark.order(before=["test_person_recognition_route.py::test_recognition_one_person_file"])
async def test_registration_one_person_file(
    test_app_asyncio, test_mysql_connec, test_redis_connec, mock_one_face_image_1_file, mock_person_data_dict
):
    """
    Test one person registration
    """
    # purge MYSQL_TEST_TABLE related cache first
    for key in test_redis_connec.keys(f"{MYSQL_TEST_TABLE}_*"):
        test_redis_connec.delete(key)
    fpath, fcontent = mock_one_face_image_1_file
    param_dict = mock_person_data_dict(TEST_PERSON_FILE_ID)
    files = [("img_file", (fpath, fcontent, "application/jpeg"))]

    response = await test_app_asyncio.post("/register_person_file", files=files, params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_FILE_ID} registered into database",
    }


@pytest.mark.asyncio
@pytest.mark.order(before=["test_person_recognition_route.py::test_recognition_one_person_url"])
async def test_registration_one_person_url(
    test_app_asyncio,
    test_mysql_connec,
    test_redis_connec,
    mock_one_face_image_2_file,
    mock_person_data_dict,
    mock_download_url_file,  # Use the fixture
):
    """
    Test one person registration via URL
    """
    # Clear cache
    for key in test_redis_connec.keys(f"{MYSQL_TEST_TABLE}_*"):
        test_redis_connec.delete(key)

    fpath, fcontent = mock_one_face_image_2_file
    param_dict = mock_person_data_dict(TEST_PERSON_URL_ID)

    with patch("routes.register_person.download_url_file", mock_download_url_file(fcontent)):
        param_dict["img_url"] = "https://example.com/test.jpg"
        response = await test_app_asyncio.post("/register_person_url", params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_URL_ID} registered into database",
    }