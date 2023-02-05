"""
Test face recognition
"""
import pytest


@pytest.mark.asyncio
async def test_recogntion_one_face(test_app_asyncio, mock_one_face_image):
    fpath, fcontent = mock_one_face_image
    files = [('file', (fpath, fcontent, 'application/jpeg'))]
    response = await test_app_asyncio.post("/recognize_face_file", files=files)

    person_name = "asd"
    assert response.status_code == 200
    assert response.json() == {"status": "success",
                               "message": f"Detected face matches {person_name}",
                               "match_name": person_name}
