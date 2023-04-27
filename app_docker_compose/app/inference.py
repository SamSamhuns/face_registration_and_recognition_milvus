"""
Inference functions for registering and recognizing face with trtserver models and save/search with milvus database server
"""
import os
import shutil

import redis
import pymysql
from pymysql.cursors import DictCursor
from pymilvus import MilvusException

from triton_server.inference_trtserver import run_inference
from api.milvus import get_milvus_connec
from api.mysql import (insert_person_data_into_sql,
                       select_person_data_from_sql_with_id,
                       delete_person_data_from_sql_with_id)
from config import (DOWNLOAD_IMAGE_PATH,
                    REDIS_HOST, REDIS_PORT,
                    MYSQL_HOST, MYSQL_PORT,
                    MYSQL_USER, MYSQL_PASSWORD,
                    MYSQL_DATABASE, MYSQL_CUR_TABLE,
                    MILVUS_HOST, MILVUS_PORT,
                    FACE_VECTOR_DIM, FACE_METRIC_TYPE,
                    FACE_INDEX_TYPE, FACE_COLLECTION_NAME)

# connect to Redis
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True)

# Connect to MySQL
mysql_conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    db=MYSQL_DATABASE,
    cursorclass=DictCursor)

# connect to milvus connec
milvus_conn = get_milvus_connec(
    collection_name=FACE_COLLECTION_NAME,
    milvus_host=MILVUS_HOST,
    milvus_port=MILVUS_PORT,
    vector_dim=FACE_VECTOR_DIM,
    metric_type=FACE_METRIC_TYPE,
    index_type=FACE_INDEX_TYPE)
# load milvus_conn into memory
milvus_conn.load()


def get_registered_person(
        person_id: int,
        table: str) -> dict:
    """
    Get registered person by person_id.
    Checks redis cache, otherwise query mysql
    """
    # try cached redis data
    redis_key = f"{table}_{person_id}"
    cached_person_dict = redis_conn.hgetall(name=redis_key)
    if cached_person_dict:
        print("record matching id: {person_id} retrieved from redis cache")
        return {"status": "success",
                "message": f"record matching id: {person_id} retrieved from redis cache",
                "person_data": cached_person_dict}

    # if cache is not found, query mysql
    return select_person_data_from_sql_with_id(
        mysql_conn, table, person_id)


def unregister_person(
        person_id: int,
        table: str) -> dict:
    """
    Deletes a registered person based on the unique person_id.
    Must use expr with the term expression `in` for delete operations
    Operation is atomic, if one delete op fails, all ops fail
    """
    try:
        # unregister from mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_del_resp = delete_person_data_from_sql_with_id(
            mysql_conn, table, person_id, commit=False)
        if mysql_del_resp["status"] == "failed":
            raise pymysql.Error

        # unregister from milvus
        expr = f'person_id in [{person_id}]'
        milvus_conn.delete(expr)
        print(
            f"Vector for person with id: {person_id} deleted from milvus db.✅️")

        # clear redis cache
        redis_key = f"{table}_{person_id}"
        redis_conn.delete(redis_key)

        # commit mysql record delete
        mysql_conn.commit()
    except (pymysql.Error, MilvusException, redis.RedisError) as excep:
        print(f"{excep}. ❌")
        return {"status": "failed",
                "message": f"person with id {person_id} couldn't be unregistered from database"}
    print(f"person record with id {person_id} unregistered from database.✅️")
    return {"status": "success",
            "message": f"person record with id {person_id} unregistered from database"}


def register_person(
        model_name: str,
        file_path: str,
        threshold: float,
        person_data: dict,
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Detects faces in image from the file_path and 
    saves the face feature vector & the related person_data dict.
    person_data dict should be based on the init.sql table schema
    Operation is atomic, if one insert op fails, all ops fail
    """
    person_id = person_data["ID"]  # uniq person id from user input
    # check if face already exists in redis/mysql
    if get_registered_person(person_id, table)["status"] == "success":
        return {"status": "failed",
                "message": f"person with id {person_id} already exists in database"}

    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        return_mode="json")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed",
                "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    try:
        # insert record into mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_insert_resp = insert_person_data_into_sql(
            mysql_conn, table, person_data, commit=False)
        if mysql_insert_resp["status"] == "failed":
            raise pymysql.Error

        # insert face_vector into milvus milvus_conn
        face_vector = pred_dict["face_feats"][0].tolist()
        data = [[face_vector], [person_id]]
        milvus_conn.insert(data)
        print(
            f"Vector for person with id: {person_id} inserted into milvus db. ✅️")
        # After final entity is inserted, it is best to call flush to have no growing segments left in memory
        milvus_conn.flush()

        # cache data in redis
        redis_key = f"{table}_{person_id}"
        person_data["birthdate"] = str(person_data["birthdate"])
        redis_conn.hset(redis_key, mapping=person_data)  # hash set data
        redis_conn.expire(redis_key, 3600)  # cache for 1 hour

        # commit mysql record insertion
        mysql_conn.commit()
    except (pymysql.Error, MilvusException, redis.RedisError) as excep:
        print(f"{excep}. ❌")
        return {"status": "failed",
                "message": f"person with id {person_id} couldn't be registered into database"}
    # save person image to volume if successfully registered
    shutil.copy(file_path, os.path.join(DOWNLOAD_IMAGE_PATH, f"{person_id}.jpg"))

    print(f"person record with id {person_id} registered into database.✅️")
    return {"status": "success",
            "message": f"person record with id {person_id} registered into database"}


def recognize_person(
        model_name: str,
        file_path: str,
        threshold: float,
        face_dist_threshold: float = 0.1,
        table: str = MYSQL_CUR_TABLE) -> dict:
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
        return {"status": "failed",
                "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    face_vector = pred_dict["face_feats"]
    # run a vector search and return the closest face with the L2 metric
    search_params = {"metric_type": "L2",  "params": {"nprobe": 2056}}
    results = milvus_conn.search(
        data=face_vector,
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["person_id"])
    if not results:
        return {"status": "failed",
                "message": "no saved face entries found in database"}

    results = sorted(results, key=lambda k: k.distances)

    face_dist = results[0].distances[0]
    person_id = results[0][0].entity.get("person_id")
    if face_dist > face_dist_threshold:
        return {"status": "success",
                "message": "no similar faces were found in the database"}

    get_person_resp = get_registered_person(person_id, table)
    if get_person_resp["status"] == "success":
        return {"status": "success",
                "message": f"detected face matches id: {person_id}",
                "person_data": get_person_resp["person_data"]}
    return get_person_resp
