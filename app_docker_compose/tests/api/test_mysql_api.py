"""
Test mysql api
The mysql server must be running in the appropriate port
"""
from tests.conftest import MYSQL_TEST_TABLE
from app.api.mysql import insert_person_data_into_sql, select_person_data_from_sql_with_id, delete_person_data_from_sql_with_id


TEST_PERSON_ID = -2  # should use a different person_id here


def test_insert_person_mysql(test_mysql_connec, mock_person_data_dict):
    resp = insert_person_data_into_sql(test_mysql_connec, MYSQL_TEST_TABLE, mock_person_data_dict(TEST_PERSON_ID))
    assert resp == {"status": "success",
                    "message": "record inserted into mysql db"}


def test_select_person_mysql(test_mysql_connec, mock_person_data_dict):
    resp = select_person_data_from_sql_with_id(test_mysql_connec, MYSQL_TEST_TABLE, TEST_PERSON_ID)
    assert resp == {"status": "success",
                    "message": f"record matching id: {TEST_PERSON_ID} retrieved from mysql db",
                    "person_data": mock_person_data_dict(TEST_PERSON_ID)}


def test_delete_person_mysql(test_mysql_connec):
    resp = delete_person_data_from_sql_with_id(test_mysql_connec, MYSQL_TEST_TABLE, TEST_PERSON_ID)
    assert resp == {"status": "success",
                    "message": "record deleted from mysql db"}