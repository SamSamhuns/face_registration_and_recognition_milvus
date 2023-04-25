"""
Test redis api
The redis server must be running in the appropriate port
"""
import copy
from app.config import MYSQL_PERSON_TABLE


def test_redis_insert_get_del_ops(test_redis_connec, mock_person_data_dict):

    person_dict = copy.deepcopy(mock_person_data_dict)
    person_dict = {k:str(v) for k, v in person_dict.items()}

    # cache data in redis
    redis_key = f"{MYSQL_PERSON_TABLE}_{person_dict['ID']}"
    test_redis_connec.hset(name=redis_key, mapping=person_dict)  # hash set data
    test_redis_connec.expire(redis_key, 3600)  # cache for 1 hour

    # retrieve cached data
    cached_dict = test_redis_connec.hgetall(name=redis_key)
    assert cached_dict == person_dict

    # delete cache
    test_redis_connec.delete(redis_key)

    # attempt retrieval again
    cached_dict = test_redis_connec.hgetall(name=redis_key)
    assert cached_dict == {}