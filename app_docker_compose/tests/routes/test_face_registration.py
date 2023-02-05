"""
Test face registration
"""
import pytest


@pytest.mark.asyncio
async def test_registration_one_face(test_app_asyncio, mock_one_face_image):
    fpath, fcontent = mock_one_face_image
    person_name = "asd"
    files = [('file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post("/register_face_file", files=files, params={'person_name': person_name})

    assert response.status_code == 200
    assert response.json() == {"status": "success",
                               "message": "face successfully saved"}
