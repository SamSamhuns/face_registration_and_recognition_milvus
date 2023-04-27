"""
Test person recognition route
"""
import copy
import pytest
from app.models import ModelType
from tests.conftest import TEST_PERSON_FILE_ID, TEST_PERSON_URL_ID


@pytest.mark.asyncio
async def test_recognition_one_person_file(
        test_app_asyncio, test_mysql_connec, mock_one_face_image_1, mock_person_data_dict):
    """
    Test one person face recognition with img file
    """
    fpath, fcontent = mock_one_face_image_1
    param_dict = copy.deepcopy(mock_person_data_dict(TEST_PERSON_FILE_ID))
    param_dict = {k: str(v) for k, v in param_dict.items()}

    files = [('img_file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post(
        "/recognize_person_file",
        files=files)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"detected face matches id: {TEST_PERSON_FILE_ID}",
        "person_data": param_dict}


@pytest.mark.asyncio
async def test_recognition_one_person_url(
        test_app_asyncio, test_mysql_connec, mock_one_face_image_2_url, mock_person_data_dict):
    """
    Test one person face recognition with img url
    """
    furl = mock_one_face_image_2_url
    param_dict = copy.deepcopy(mock_person_data_dict(TEST_PERSON_URL_ID))
    param_dict = {k: str(v) for k, v in param_dict.items()}
    param_dict["img_url"] = furl
    param_dict["model_type"] = ModelType.SLOW

    response = await test_app_asyncio.post(
        "/recognize_person_url",
        params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"detected face matches id: {TEST_PERSON_URL_ID}",
        "person_data": param_dict}
