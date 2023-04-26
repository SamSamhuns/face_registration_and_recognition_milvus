"""
Test person recognition route
"""
import copy
import pytest
from tests.conftest import TEST_PERSON_ID


# TODO add url check


@pytest.mark.asyncio
async def test_recognition_one_person(test_app_asyncio, test_mysql_connec, mock_one_face_image, mock_person_data_dict):
    fpath, fcontent = mock_one_face_image
    param_dict = copy.deepcopy(mock_person_data_dict(TEST_PERSON_ID))
    param_dict = {k:str(v) for k, v in param_dict.items()}

    files = [('img_file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post("/recognize_person_file", files=files)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"detected face matches id: {TEST_PERSON_ID}",
        "person_data": param_dict}
