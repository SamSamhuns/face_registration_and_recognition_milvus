"""
Test person registration route
"""
import pytest
from tests.conftest import TEST_PERSON_ID, MYSQL_TEST_TABLE


# TODO add url check


@pytest.mark.asyncio
@pytest.mark.order(before=["test_person_recognition_route.py::test_recognition_one_person"])
async def test_registration_one_person(test_app_asyncio, test_mysql_connec, test_redis_connec, mock_one_face_image, mock_person_data_dict):
    """
    Test one person registration
    """
    # purge MYSQL_TEST_TABLE related cache first
    for key in test_redis_connec.keys(f"{MYSQL_TEST_TABLE}_*"):
        test_redis_connec.delete(key)
    fpath, fcontent = mock_one_face_image
    param_dict = mock_person_data_dict(TEST_PERSON_ID)

    files = [('img_file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post(
        "/register_person_file",
        files=files,
        params=param_dict)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": f"person record with id {TEST_PERSON_ID} registered into database"}
