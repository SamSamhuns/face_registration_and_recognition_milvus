"""
Test person recognition route
"""
import copy
from unittest.mock import patch

import pytest
from tests.conftest import TEST_PERSON_FILE_ID, TEST_PERSON_URL_ID


@pytest.mark.asyncio
async def test_recognition_one_person_file(
    test_app_asyncio, test_mysql_connec, mock_one_face_image_1_file, mock_person_data_dict
):
    """
    Test one person face recognition with img file
    """
    fpath, fcontent = mock_one_face_image_1_file
    param_dict = copy.deepcopy(mock_person_data_dict(TEST_PERSON_FILE_ID))
    param_dict = {k: str(v) for k, v in param_dict.items()}

    files = [("img_file", (fpath, fcontent, "application/jpeg"))]
    response = await test_app_asyncio.post("/recognize_person_file", files=files)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"detected face matches id: {TEST_PERSON_FILE_ID}",
        "person_data": param_dict,
    }

@pytest.mark.asyncio
async def test_recognition_one_person_url(
    test_app_asyncio,
    test_mysql_connec,
    mock_one_face_image_2_file,
    mock_person_data_dict,
    mock_download_url_file,  # Use the fixture
):
    """
    Test one person face recognition with img url
    """
    fpath, fcontent = mock_one_face_image_2_file
    param_dict = copy.deepcopy(mock_person_data_dict(TEST_PERSON_URL_ID))
    param_dict = {k: str(v) for k, v in param_dict.items()}

    # Patch the download function
    with patch("routes.recognize_person.download_url_file", mock_download_url_file(fcontent)):
        param_dict["img_url"] = "https://example.com/test.jpg"
        response = await test_app_asyncio.post("/recognize_person_url", params=param_dict)

    assert response.status_code == 200
    del param_dict["img_url"]
    assert response.json() == {
        "status": "success",
        "message": f"detected face matches id: {TEST_PERSON_URL_ID}",
        "person_data": param_dict,
    }
