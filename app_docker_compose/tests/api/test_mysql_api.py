"""
Test mysql api
The mysql server must be running in the appropriate port
"""
from app.config import MYSQL_PERSON_TABLE
from app.api.mysql import insert_person_data_into_sql, select_person_data_from_sql_with_id, delete_person_data_from_sql_with_id


def test_insert_person_mysql(test_mysql_connec, mock_person_data_dict):
    resp = insert_person_data_into_sql(test_mysql_connec, MYSQL_PERSON_TABLE, mock_person_data_dict)
    assert resp == {"status": "success",
                    "message": "record inserted into mysql db"}


def test_select_person_mysql(test_mysql_connec, mock_person_data_dict):
    person_id = 0
    resp = select_person_data_from_sql_with_id(test_mysql_connec, MYSQL_PERSON_TABLE, person_id)
    assert resp == {"status": "success",
                    "message": f"record matching id: {person_id} retrieved from mysql db",
                    "person_data": mock_person_data_dict}


def test_delete_person_mysql(test_mysql_connec):
    person_id = 0
    resp = delete_person_data_from_sql_with_id(test_mysql_connec, MYSQL_PERSON_TABLE, person_id)
    assert resp == {"status": "success",
                    "message": "record deleted from mysql db"}