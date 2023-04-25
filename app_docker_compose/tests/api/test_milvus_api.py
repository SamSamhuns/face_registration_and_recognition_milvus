"""
Test milvus api
The milvus server must be running in the appropriate port 
"""
import numpy as np
from app.config import FACE_VECTOR_DIM


def test_insert_person_milvus(test_milvus_connec):
    person_id = 0
    emb_vec = [0.0] * FACE_VECTOR_DIM
    data = [[emb_vec], [person_id]]
    test_milvus_connec.insert(data)


def test_get_person_milvus(test_milvus_connec):
    person_id = 0
    emb_vec = [0.0] * FACE_VECTOR_DIM
    expr = f'person_id == {person_id}'
    results = test_milvus_connec.query(
        expr=expr,
        offset=0,
        limit=10,
        output_fields=["person_id", "embedding"],
        consistency_level="Strong")
    
    person_id = results[0]["person_id"]
    emb = results[0]["embedding"]
    assert emb == emb_vec


def test_delete_person_milvus(test_milvus_connec):
    person_id = 0
    expr = f'person_id in [{person_id}]'
    test_milvus_connec.delete(expr)
