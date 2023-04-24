import pymysql


def insert_person_data_into_sql(mysql_conn, mysql_tb, person_data: dict, commit: bool = True) -> dict:
    """
    Insert person_data into mysql table with param binding
    Note: the transaction must be commited after if commit is False
    """
    # query fmt: `INSERT INTO mysql_tb (id, col1_name, col2_name) VALUES (%s, %s, %s)`
    query = (f"INSERT INTO {mysql_tb}" +
             f" {tuple(person_data.keys())}" +
             f" VALUES ({', '.join(['%s'] * len(person_data))})").replace("'", '')
    values = tuple(person_data.values())
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            if commit:
                mysql_conn.commit()
                print(f"record inserted into mysql db.✅️")
                return {"status": "success",
                        "message": "record inserted into mysql db"}
            print("record insertion waiting to be commit to mysql db.🕓")
            return {"status": "success", 
                    "message": "record insertion waiting to be commit to mysql db."}
    except pymysql.Error as e:
        print(f"mysql record insert failed ❌. {e}")
        return {"status": "failed",
                "message": "mysql record insertion error"}


def select_person_data_from_sql_with_id(mysql_conn, mysql_tb, person_id: int) -> dict:
    """
    Query mysql db to get full person data using the uniq person_id
    """
    query = (f"SELECT * FROM {mysql_tb} WHERE id = %s")
    values = (person_id)
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            person_data = cursor.fetchone()
            if person_data is None:
                print(f"mysql record with id: {person_id} does not exist ❌.")
                return {"status": "failed",
                        "message": "mysql record with id: {person_id} does not exist"}
            print(f"Person with id: {person_id} retrieved from mysql db.✅️")
            return {"status": "success",
                    "message": f"record matching id: {person_id} retrieved from mysql db",
                    "person_data": person_data}
    except pymysql.Error as e:
        print(f"mysql record retrieval failed ❌. {e}")
        return {"status": "failed",
                "message": "mysql record retrieval error"}


def delete_person_data_from_sql_with_id(mysql_conn, mysql_tb, person_id: int, commit: bool = True) -> dict:
    """
    Delete record from mysql db using the uniq person_id
    """
    select_query = f"SELECT * FROM {mysql_tb} WHERE id = %s"
    del_query = f"DELETE FROM {mysql_tb} WHERE id = %s"
    try:
        with mysql_conn.cursor() as cursor:
            # check if record exists in db or not
            cursor.execute(select_query, (person_id))
            if not cursor.fetchone():
                print(f"Person with id: {person_id} does not exist in mysql db.❌")
                return {"status": "failed",
                        "message": f"mysql record with id: {person_id} does not exist in db"}

            cursor.execute(del_query, (person_id))
            if commit:
                mysql_conn.commit()
                print(f"Person with id: {person_id} deleted from mysql db.✅️")
                return {"status": "success",
                        "message": "record deleted from mysql db"}
            print("record deletion waiting to be commited to mysql db.🕓")
            return {"status": "success",
                    "message": "record deletion waiting to be commited to mysql db."}
    except pymysql.Error as e:
        print(f"mysql record deletion failed ❌. {e}")
        return {"status": "failed",
                "message": "mysql record deletion error"}


if __name__ == "__main__":
    person_id = 123
    person_data = {
        "id": person_id,
        "name": "abc",
        "birthdate": "2000-01-30",
        "country": "abcd",
        "city": "abcde",
        "title": "abcdef",
        "org": "abcdefg",
    }

    from pymysql.cursors import DictCursor
    from app.config import (
        MYSQL_HOST, MYSQL_PORT,
        MYSQL_USER, MYSQL_PASSWORD,
        MYSQL_DATABASE, MYSQL_PERSON_TABLE)

    # Connect to MySQL
    mysql_conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DATABASE,
        cursorclass=DictCursor)

    try:
        mysql_insert_resp = insert_person_data_into_sql(
            mysql_conn, MYSQL_PERSON_TABLE, person_data, commit=False)
        print(mysql_insert_resp)
        raise Exception("mysql data should not be inserted")
        # Commit the transaction
        mysql_conn.commit()
    except Exception as e:
        # Rollback the transaction if an exception occurs
        # mysql_conn.rollback()
        print("Error: ROLLBACK", e)
    # finally:
    #     # Close the connection
    #     mysql_conn.close()


