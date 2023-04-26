"""
Test milvus api
The milvus server must be running in the appropriate port 
"""
import pytest
from tests.conftest import FACE_VECTOR_DIM, TEST_PERSON_ID


@pytest.mark.order(before="test_get_person_milvus")
def test_insert_person_milvus(test_milvus_connec):
    emb_vec = [0.0] * FACE_VECTOR_DIM
    data = [[emb_vec], [TEST_PERSON_ID]]
    assert test_milvus_connec.insert(data).insert_count == 1


@pytest.mark.order(before="test_delete_person_milvus")
def test_get_person_milvus(test_milvus_connec):
    emb_vec = [0.0] * FACE_VECTOR_DIM
    expr = f'person_id == {TEST_PERSON_ID}'
    results = test_milvus_connec.query(
        expr=expr,
        offset=0,
        limit=10,
        output_fields=["person_id", "embedding"],
        consistency_level="Strong")
    person_id = results[0]["person_id"]
    embedding = results[0]["embedding"]
    assert person_id == TEST_PERSON_ID, embedding == emb_vec


def test_delete_person_milvus(test_milvus_connec):
    expr = f'person_id in [{TEST_PERSON_ID}]'
    assert test_milvus_connec.delete(expr).delete_count == 1

