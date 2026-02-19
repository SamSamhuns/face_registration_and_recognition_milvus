"""
Inference functions for registering and recognizing face with trtserver models and save/search with milvus database server
"""

import logging
import os
import shutil
import threading
import time

import pymysql
import redis
from pymilvus import MilvusException
from pymysql.cursors import DictCursor

from app.api.milvus import get_milvus_collec_conn
from app.api.mysql import (
    delete_person_data_from_sql_with_id,
    insert_person_data_into_sql,
    select_all_person_data_from_sql,
    select_person_data_from_sql_with_id,
)
from app.config import (
    DOWNLOAD_IMAGE_PATH,
    FACE_COLLECTION_NAME,
    FACE_INDEX_NLIST,
    FACE_INDEX_TYPE,
    FACE_METRIC_TYPE,
    FACE_SEARCH_NPROBE,
    FACE_VECTOR_DIM,
    MILVUS_HOST,
    MILVUS_PORT,
    MYSQL_CUR_TABLE,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
    REDIS_HOST,
    REDIS_PORT,
    TRITON_SERVER_HOST,
    TRITON_SERVER_PORT,
)
from app.triton_server.inference_trtserver import run_inference

logger = logging.getLogger("inference_api")

_conn_lock = threading.Lock()
redis_conn = None
mysql_conn = None
milvus_collec_conn = None


def _retry(action, label: str, retries: int = 10, delay: float = 1.0, backoff: float = 1.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return action()
        except Exception as exc:
            last_exc = exc
            logger.warning("%s connection attempt %s/%s failed: %s", label, attempt, retries, exc)
            time.sleep(delay)
            delay *= backoff
    raise last_exc


def _connect_redis():
    conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    conn.ping()
    return conn


def _connect_mysql():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=DictCursor,
        connect_timeout=5,
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1")
    return conn


def _connect_milvus():
    conn = get_milvus_collec_conn(
        collection_name=FACE_COLLECTION_NAME,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        vector_dim=FACE_VECTOR_DIM,
        metric_type=FACE_METRIC_TYPE,
        index_type=FACE_INDEX_TYPE,
        index_metric_params={"nlist": FACE_INDEX_NLIST},
    )
    conn.load()
    return conn


def init_connections(retries: int = 10, delay: float = 1.0, backoff: float = 1.5) -> None:
    """
    Initialize Redis/MySQL/Milvus connections with retry/backoff.
    Safe to call multiple times.
    """
    global redis_conn, mysql_conn, milvus_collec_conn
    with _conn_lock:
        if redis_conn is None:
            redis_conn = _retry(_connect_redis, "redis", retries, delay, backoff)
        if mysql_conn is None:
            mysql_conn = _retry(_connect_mysql, "mysql", retries, delay, backoff)
        if milvus_collec_conn is None:
            milvus_collec_conn = _retry(_connect_milvus, "milvus", retries, delay, backoff)


def ensure_connections() -> None:
    if redis_conn is None or mysql_conn is None or milvus_collec_conn is None:
        init_connections()


def close_connections() -> None:
    global redis_conn, mysql_conn, milvus_collec_conn
    with _conn_lock:
        if redis_conn is not None:
            try:
                redis_conn.close()
            except Exception:
                pass
        redis_conn = None
        if mysql_conn is not None:
            try:
                mysql_conn.close()
            except Exception:
                pass
        mysql_conn = None
        try:
            from pymilvus import connections

            connections.disconnect("default")
        except Exception:
            pass
        milvus_collec_conn = None


def get_registered_person(person_id: int, table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Get registered person by person_id.
    Checks redis cache, otherwise query mysql
    """
    ensure_connections()
    # try cached redis data
    redis_key = f"{table}_{person_id}"
    cached_person_dict = redis_conn.hgetall(name=redis_key)
    if cached_person_dict:
        logger.info("record matching id: {person_id} retrieved from redis cache")
        return {
            "status": "success",
            "message": f"record matching id: {person_id} retrieved from redis cache",
            "person_data": cached_person_dict,
        }

    # if cache is not found, query mysql
    return select_person_data_from_sql_with_id(mysql_conn, table, person_id)


def get_all_registered_person(table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Get all registered persons query mysql
    """
    ensure_connections()
    return select_all_person_data_from_sql(mysql_conn, table)


def unregister_person(person_id: int, table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Deletes a registered person based on the unique person_id.
    Must use expr with the term expression `in` for delete operations
    Operation is atomic, if one delete op fails, all ops fail
    """
    ensure_connections()
    try:
        # unregister from mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_del_resp = delete_person_data_from_sql_with_id(mysql_conn, table, person_id, commit=False)
        if mysql_del_resp["status"] == "failed":
            raise pymysql.Error

        # unregister from milvus
        expr = f"person_id in [{person_id}]"
        milvus_collec_conn.delete(expr)
        logger.info("Vector for person with id: %s deleted from milvus db.✅️", person_id)

        # clear redis cache
        redis_key = f"{table}_{person_id}"
        redis_conn.delete(redis_key)

        # commit mysql record delete
        mysql_conn.commit()
    except (pymysql.Error, MilvusException, redis.RedisError) as excep:
        msg = f"person with id {person_id} couldn't be unregistered from database ❌"
        logger.error("%s: %s", excep, msg)
        return {"status": "failed", "message": msg}
    logger.info("person record with id %s unregistered from database.✅️", person_id)
    return {"status": "success", "message": f"person record with id {person_id} unregistered from database"}


def register_person(
    model_name: str, file_path: str, face_det_threshold: float, person_data: dict, table: str = MYSQL_CUR_TABLE
) -> dict:
    """
    Detects faces in image from the file_path and
    saves the face feature vector & the related person_data dict.
    person_data dict should be based on the init.sql table schema
    Operation is atomic, if one insert op fails, all ops fail
    """
    ensure_connections()
    person_id = person_data["ID"]  # uniq person id from user input
    # check if face already exists in redis/mysql
    if get_registered_person(person_id, table)["status"] == "success":
        return {"status": "failed", "message": f"person with id {person_id} already exists in database"}

    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=face_det_threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        host=TRITON_SERVER_HOST,
        port=TRITON_SERVER_PORT,
        return_mode="json",
    )

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    try:
        # insert record into mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_insert_resp = insert_person_data_into_sql(mysql_conn, table, person_data, commit=False)
        if mysql_insert_resp["status"] == "failed":
            raise pymysql.Error

        # insert face_vector into milvus milvus_collec_conn
        face_vector = pred_dict["face_feats"][0].tolist()
        data = [[person_id], [face_vector]]
        milvus_collec_conn.insert(data)
        logger.info("Vector for person with id: %s inserted into milvus db. ✅️", person_id)
        # After final entity is inserted, it is best to call flush to have no growing segments left in memory
        # flushes collection data from memory to storage
        milvus_collec_conn.flush()

        # cache data in redis
        redis_key = f"{table}_{person_id}"
        person_data["birthdate"] = str(person_data["birthdate"])  # redis can't ingest type date
        redis_conn.hset(redis_key, mapping=person_data)  # hash set data
        redis_conn.expire(redis_key, 3600)  # cache for 1 hour

        # commit mysql record insertion
        mysql_conn.commit()
    except (pymysql.Error, MilvusException, redis.RedisError) as excep:
        msg = f"person with id {person_id} couldn't be registered into database ❌"
        logger.error("%s: %s", excep, msg)
        return {"status": "failed", "message": msg}
    # save person image to volume if successfully registered
    shutil.copy(file_path, os.path.join(DOWNLOAD_IMAGE_PATH, f"{person_id}.jpg"))

    logger.info("person record with id %s registered into database.✅️", person_id)
    return {"status": "success", "message": f"person record with id {person_id} registered into database"}


def recognize_person(
    model_name: str,
    file_path: str,
    face_det_threshold: float,
    face_dist_threshold: float = 0.1,
    table: str = MYSQL_CUR_TABLE,
) -> dict:
    """
    Detects faces in image from the file_path and finds the most similar face vector
    from a set of saved face vectors
    """
    ensure_connections()
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=face_det_threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        host=TRITON_SERVER_HOST,
        port=TRITON_SERVER_PORT,
        return_mode="json",
    )

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    face_vector = pred_dict["face_feats"]
    # run a vector search and return the closest face with the L2 metric
    search_params = {"metric_type": "L2", "params": {"nprobe": FACE_SEARCH_NPROBE}}
    results = milvus_collec_conn.search(
        data=face_vector, anns_field="embedding", param=search_params, limit=3, output_fields=["person_id"]
    )
    if not results:
        return {"status": "failed", "message": "no saved face entries found in database"}

    results = sorted(results, key=lambda k: k.distances)

    face_dist = results[0].distances[0]

    person_id = results[0][0].entity.get("person_id")
    if face_dist > face_dist_threshold:
        return {"status": "success", "message": "no similar faces were found in the database"}

    get_person_resp = get_registered_person(person_id, table)
    if get_person_resp["status"] == "success":
        return {
            "status": "success",
            "message": f"detected face matches id: {person_id}",
            "person_data": get_person_resp["person_data"],
        }
    return get_person_resp
