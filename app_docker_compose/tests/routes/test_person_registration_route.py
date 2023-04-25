"""
Test person registration route
"""
import pytest
import pymysql
from app.api.mysql import select_person_data_from_sql_with_id
from app.inference import unregister_person
from tests.conftest import TEST_PERSON_ID, MYSQL_PERSON_TABLE


# TODO add url check


@pytest.mark.asyncio
async def test_registration_one_person(test_app_asyncio, test_mysql_connec, mock_one_face_image, mock_person_data_dict):
    fpath, fcontent = mock_one_face_image

    if select_person_data_from_sql_with_id(test_mysql_connec, MYSQL_PERSON_TABLE, TEST_PERSON_ID)["status"] == "success":
        unregister_person(TEST_PERSON_ID)
    param_dict = mock_person_data_dict(TEST_PERSON_ID)
    param_dict["ID"] = TEST_PERSON_ID

    files = [('img_file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post(
        "/register_person_file",
        files=files,
        params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_ID} registered into database"}
