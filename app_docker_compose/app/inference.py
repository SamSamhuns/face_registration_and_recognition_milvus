"""
Inference functions for registering and recognizing face with trtserver models and save/search with milvus database server
"""
import os

import redis
import pymysql
from pymilvus import connections, MilvusException
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

from triton_server.inference_trtserver import run_inference


_REDIS_HOST = os.getenv("REDIS_HOST", default="127.0.0.1")
_REDIS_PORT = os.getenv("REDIS_PORT", default="6379")

_MYSQL_HOST = os.getenv("MYSQL_HOST", default="127.0.0.1")
_MYSQL_PORT = os.getenv("MYSQL_PORT", default="3306")
_MYSQL_USER = os.getenv("MYSQL_USER", default="user")
_MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", default="pass")
_MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", default="default")

_MILVUS_HOST = os.getenv("MILVUS_HOST", default="127.0.0.1")
_MILVUS_PORT = os.getenv("MILVUS_PORT", default="19530")
_VECTOR_DIM = 128
_METRIC_TYPE = "L2"
_INDEX_TYPE = "IVF_FLAT"
COLLECTION_NAME = 'faces'

# connect to Redis
redis_conn = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT)

# Connect to MySQL
mysql_conn = pymysql.connect(
    host=_MYSQL_HOST,
    port=_MYSQL_PORT,
    user=_MYSQL_USER,
    password=_MYSQL_PASSWORD,
    db=_MYSQL_DATABASE
)

# connect to milvus server
connections.connect(alias="default", host=_MILVUS_HOST, port=_MILVUS_PORT)

# load collection
# if collection is not present create one
if not utility.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    descrition="ids", is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                    descrition="embedding vectors", dim=_VECTOR_DIM),
        FieldSchema(name="name", dtype=DataType.VARCHAR,
                    descrition="persons name", max_length=200)
    ]
    schema = CollectionSchema(
        fields=fields, description='face recognition system')
    collection = Collection(name=COLLECTION_NAME,
                            consistency_level="Strong",
                            schema=schema, using='default')
    print(f"Collection {COLLECTION_NAME} created.✅️")

    # Indexing the collection
    print("Indexing the Collection...🔢️")
    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': _METRIC_TYPE,
        'index_type': _INDEX_TYPE,
        'params': {"nlist": 4096}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection {COLLECTION_NAME} indexed.✅️")
else:
    print(f"Collection {COLLECTION_NAME} present already.✅️")
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

    face_vector = pred_dict["face_feats"][0].tolist()
    data = [[face_vector], [person_name]]

    # insert data into collection
    collection.insert(data)
    print(f"Vector for {person_name} inserted in.✅️")
    # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    collection.flush()

    return {"status": "success", "message": "face successfully saved"}


def unregister_face(person_name: str) -> dict:
    """
    Deletes a registered face based on the name.
    Recommended to switch to using person id instead
    Must use expr with the term expression `in` for delete operations
    """
    expr = f'name in ["{person_name}"]'

    try:
        collection.delete(expr)
        print(f"Person {person_name} unregistered from database.✅️")
        return {"status": "success", "message": f"Person {person_name} unregistered"}
    except MilvusException as excep:
        print(excep)
        return {"status": "failure", "message": f"Person {person_name} couldn't be unregistered"}


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

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    face_vector = pred_dict["face_feats"]
    # run a vector search and return the closest face with the L2 metric
    search_params = {"metric_type": "L2",  "params": {"nprobe": 2056}}
    results = collection.search(
        data=face_vector,
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["name"])
    if not results:
        return {"status": "failure", "message": "No saved face entries found in database"}

    results = sorted(results, key=lambda k: k.distances)

    face_dist = results[0].distances[0]
    face_name = results[0][0].entity.get("name")
    if face_dist > face_dist_threshold:
        return {"status": "success", "message": "No similar faces were found in the database"}

    return {"status": "success", "message": f"Detected face matches {face_name}", "match_name": face_name}


def get_registered_face(person_name: str) -> dict:
    """
    Get registered face by person_name.
    """
    failure_msg = f"Person {person_name} not found in database"
    expr = f'name == "{person_name}"'
    try:
        results = collection.query(
            expr=expr,
            offset=0,
            limit=10,
            output_fields=["name"],
            consistency_level="Strong")
        if not results:
            return {"status": "failure", "message": failure_msg}

        found_person_name = results[0]["name"]
        print(f"Person {found_person_name} found in database.✅️")
        return {"status": "success", "message": f"Person {found_person_name} found"}
    except MilvusException as excep:
        print(excep)
        return {"status": "failure", "message": failure_msg}
