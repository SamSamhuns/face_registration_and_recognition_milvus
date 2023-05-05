"""
Basic operations for faces
"""
from fastapi import APIRouter

from inference import get_registered_person as get_registered_person_api
from inference import unregister_person as unregister_person_api


router = APIRouter()

# note: person insert/post is done with person_registration route instead


@router.get("/person/{person_id}")
async def get_registered_person(person_id: int):
    """Gets the registered person with the given ID."""
    return get_registered_person_api(person_id)


@router.delete("/person/{person_id}")
async def unregister_person(person_id: int):
    """Unregisters the person with the given ID."""
    return unregister_person_api(person_id)
