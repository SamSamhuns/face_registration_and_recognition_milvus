"""
Basic operations for faces
"""

from fastapi import APIRouter, HTTPException, status
from inference import (
    get_all_registered_person as get_all_registered_person_api,
)
from inference import (
    get_registered_person as get_registered_person_api,
)
from inference import unregister_person as unregister_person_api

router = APIRouter()

# note: person insert/post is done with person_registration route instead


@router.get("/person")
async def get_all_registered_persons():
    """Gets all registered persons."""
    response_data = {}
    try:
        response_data = get_all_registered_person_api()
    except Exception as excep:
        response_data["message"] = "No registered persons found"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep
    return response_data


@router.get("/person/{person_id}")
async def get_registered_person(person_id: int):
    """Gets the registered person with the given ID."""
    response_data = {}
    try:
        response_data = get_registered_person_api(person_id)
    except Exception as excep:
        response_data["message"] = f"failed to get the registered person with id {person_id}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep
    return response_data


@router.delete("/person/{person_id}")
async def unregister_person(person_id: int):
    """Unregisters the person with the given ID."""
    response_data = {}
    try:
        response_data = unregister_person_api(person_id)
    except Exception as excep:
        response_data["message"] = f"failed to unregistered person with id {person_id}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data) from excep
    return response_data
