"""
Test mysql api
The mysql server must be running in the appropriate port
"""
import pytest
from tests.conftest import MYSQL_TEST_TABLE, TEST_PERSON_MYSQL_ID
from app.api.mysql import (insert_person_data_into_sql,
                           select_all_person_data_from_sql,
                           select_person_data_from_sql_with_id,
                           delete_person_data_from_sql_with_id)


@pytest.mark.order(before="test_select_person_mysql")
def test_insert_person_mysql(test_mysql_connec, mock_person_data_dict):
    """Inserts test person data into MySQL database."""
    resp = insert_person_data_into_sql(
        test_mysql_connec, MYSQL_TEST_TABLE, mock_person_data_dict(TEST_PERSON_MYSQL_ID))
    assert resp == {"status": "success",
                    "message": "record inserted into mysql db"}


@pytest.mark.order(before="test_delete_person_mysql")
def test_select_person_mysql(test_mysql_connec, mock_person_data_dict):
    """Selects test person data from MySQL database."""
    resp = select_person_data_from_sql_with_id(
        test_mysql_connec, MYSQL_TEST_TABLE, TEST_PERSON_MYSQL_ID)
    assert resp == {"status": "success",
                    "message": f"record matching id: {TEST_PERSON_MYSQL_ID} retrieved from mysql db",
                    "person_data": mock_person_data_dict(TEST_PERSON_MYSQL_ID)}


@pytest.mark.order(before="test_delete_person_mysql")
def test_select_all_person_mysql(test_mysql_connec, mock_person_data_dict):
    """Selects test person data from MySQL database."""
    resp = select_all_person_data_from_sql(
        test_mysql_connec, MYSQL_TEST_TABLE)
    assert resp == {"status": "success",
                    "message": "All person records retrieved from mysql db",
                    "person_data": [mock_person_data_dict(TEST_PERSON_MYSQL_ID)]}


def test_delete_person_mysql(test_mysql_connec):
    """Deletes test person data from MySQL database."""
    resp = delete_person_data_from_sql_with_id(
        test_mysql_connec, MYSQL_TEST_TABLE, TEST_PERSON_MYSQL_ID)
    assert resp == {"status": "success",
                    "message": "record deleted from mysql db"}
