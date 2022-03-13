from fastapi import APIRouter

router = APIRouter()


@router.get("/face")
def get_registered_face():
    return {"TODO: Should return all registered faces and corresponding assigned name"}


@router.delete("/face")
def delete_registered_face():
    return {"TODO: Should delete a registered face and corresponding assigned name"}
