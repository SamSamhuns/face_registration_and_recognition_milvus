"""
Inference functions
"""
import os

import numpy as np
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

from triton_server.inference_trtserver import run_inference


_MILVUS_HOST = os.getenv("MILVUS_HOST", default = "127.0.0.1") 
_MILVUS_PORT = os.getenv("MILVUS_PORT", default = "19530")
_VECTOR_DIM = 128
COLLECTION_NAME = 'faces'


# create connec to milvus server
connections.connect(alias="default", host=_MILVUS_HOST, port=_MILVUS_PORT)

# load collection
# if collection is not present create one
if not utility.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, descrition="ids", is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="embedding vectors", dim=_VECTOR_DIM),
        FieldSchema(name="name", dtype=DataType.VARCHAR, descrition="persons name", max_length=200)
    ]
    schema = CollectionSchema(fields=fields, description='face recognition system')
    collection = Collection(name=COLLECTION_NAME, schema=schema, using='default')
    print(f"Collection {COLLECTION_NAME} created.✅️")

    # Indexing the collection
    print("Indexing the Collection...🔢️")
    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":4096}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection {COLLECTION_NAME} indexed.✅️")
else:
    print("Collection present already.")
    collection = Collection(COLLECTION_NAME)


# load collection into memory
collection.load()


def register_face(model_name: str,
                  file_path: str,
                  threshold: float,
                  person_name: str) -> dict:
    """
    Detects faces in image from the file_path and saves the face feature vector.
    """
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        return_mode="json")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    # save face feature vector to local system for now
    face_vector = pred_dict["face_feats"][0].tolist()
    data = [[face_vector], [person_name]]

    # insert data into collection book
    collection.insert(data)
    print("Vector inserted in.✅️")
    # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    collection.flush()

    return {"status": "success", "message": "face successfully saved"}


def recognize_face(model_name: str,
                   file_path: str,
                   threshold: float,
                   face_dist_threshold: float = 0.1) -> dict:
    """
    Detects faces in image from the file_path and finds the most similar face vector 
    from a set of saved face vectors
    """
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        return_mode="json")
    print(pred_dict)
    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    face_vector = pred_dict["face_feats"]
    # find and return the closest face
    search_params = {"metric_type": "L2",  "params": {"nprobe": 2056}}
    results = collection.search(data=face_vector, anns_field="embedding", param=search_params, limit=3)
    results = sorted(results, key=lambda k: k.distances)
    ref_id = results[0].ids[0]
    ref_dist = results[0].distances[0]

    matched_data = collection.query(
        expr = f"id == {ref_id}",
        offset = 0,
        limit = 1, 
        output_fields = ["id", "name"],
        consistency_level="Strong"
    )

    # a threshold should also be added so that a new completely new face is reported as non-present in the database
    closest_face = matched_data[0]["name"]

    return {"status": "success", "message": f"Detected face matches {closest_face}", "match_name": closest_face}
