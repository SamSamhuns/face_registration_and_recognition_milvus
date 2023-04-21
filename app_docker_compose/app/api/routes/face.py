"""
Basic operations for faces
"""
from fastapi import APIRouter
from inference import get_registered_face as get_registered_face_api
from inference import unregister_face as unregister_face_api


router = APIRouter()


@router.get("/face")
def get_all_registered_face():
    return {"TODO: Should return all registered faces and corresponding assigned name"}


@router.get("/face/{person_id}")
def get_registered_face(person_id: int):
    return get_registered_face_api(person_id)


@router.delete("/face/{person_id}")
def unregister_face(person_id: int):
    return unregister_face_api(person_id)
