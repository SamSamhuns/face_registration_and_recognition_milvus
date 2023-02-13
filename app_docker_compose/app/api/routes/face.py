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


@router.get("/face/{face_name}")
def get_registered_face(person_name: str):
    return get_registered_face_api(person_name)


@router.delete("/face/{face_name}")
def unregister_face(person_name: str):
    return unregister_face_api(person_name)
