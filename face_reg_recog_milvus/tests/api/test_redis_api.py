"""
Test redis api
The redis server must be running in the appropriate port
"""

import copy

import pytest
from tests.conftest import MYSQL_TEST_TABLE


@pytest.mark.order(before="test_redis_get_ops")
def test_redis_insert_get_del_ops(test_redis_connec, mock_person_data_dict):
    """Inserts test data into Redis, retrieves it, and then deletes it."""
    person_dict = copy.deepcopy(mock_person_data_dict())
    person_dict = {k: str(v) for k, v in person_dict.items()}
    redis_key = f"{MYSQL_TEST_TABLE}_{person_dict['ID']}"

    # cache data in redis
    test_redis_connec.hset(name=redis_key, mapping=person_dict)  # hash set data
    test_redis_connec.expire(redis_key, 3600)  # cache for 1 hour


@pytest.mark.order(before="test_redis_del_ops")
def test_redis_get_ops(test_redis_connec, mock_person_data_dict):
    """Retrieves test data from Redis."""
    person_dict = copy.deepcopy(mock_person_data_dict())
    person_dict = {k: str(v) for k, v in person_dict.items()}
    redis_key = f"{MYSQL_TEST_TABLE}_{person_dict['ID']}"

    # retrieve cached data
    cached_dict = test_redis_connec.hgetall(name=redis_key)
    assert cached_dict == person_dict


def test_redis_del_ops(test_redis_connec, mock_person_data_dict):
    """Deletes test data from Redis."""
    person_dict = copy.deepcopy(mock_person_data_dict())
    person_dict = {k: str(v) for k, v in person_dict.items()}
    redis_key = f"{MYSQL_TEST_TABLE}_{person_dict['ID']}"

    # delete cache
    assert test_redis_connec.delete(redis_key) == 1

    # attempt retrieval again
    cached_dict = test_redis_connec.hgetall(name=redis_key)
    assert cached_dict == {}
