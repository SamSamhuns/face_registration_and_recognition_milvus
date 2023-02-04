from fastapi import APIRouter


router = APIRouter()


@router.get("/face")
def get_all_registered_face():
    return {"TODO: Should return all registered faces and corresponding assigned name"}


@router.get("/face/{face_name}")
def get_registered_face(face_name: str):
    return {f"TODO: Should return the registered faces based on the unique {face_name}"}


@router.delete("/face/{face_name}")
def delete_registered_face(face_name: str):
    return {f"TODO: Should delete a registered face based on the unique {face_name}"}
