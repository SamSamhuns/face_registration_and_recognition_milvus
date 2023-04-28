"""
milvus api functions
"""
from typing import List
from pymilvus import connections, MilvusException
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility


def get_milvus_connec(
        collection_name: str,
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        vector_dim: int = 128,
        metric_type: str = "L2",
        index_type: str = "IVF_FLAT",
        index_metric_params: dict = None):
    """
    Gets the milvus connection with the given collection name otherwise creates a new one
    Note: index_metric_params: dict = {"nlist": 4096} for index_type == "IVF_FLAT"
    """
    # connect to milvus
    connections.connect(
        alias="default",
        host=milvus_host,
        port=milvus_port)

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        descrition="ids", is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        descrition="embedding vectors", dim=vector_dim),
            FieldSchema(name="person_id", dtype=DataType.INT64,
                        descrition="persons unique id")
        ]
        schema = CollectionSchema(
            fields=fields, description='face recognition system')
        milvus_conn = Collection(name=collection_name,
                                 consistency_level="Strong",
                                 schema=schema, using='default')
        print(f"Collection {collection_name} created.✅️")

        # Indexing the milvus_conn
        print("Indexing the Collection...🕓")
        # create IVF_FLAT index for milvus_conn.
        index_params = {
            'metric_type': metric_type,
            'index_type': index_type,
            'params': index_metric_params
        }
        milvus_conn.create_index(
            field_name="embedding", index_params=index_params)
        print(f"Collection {collection_name} indexed.✅️")
    else:
        print(f"Collection {collection_name} present already.✅️")
        milvus_conn = Collection(collection_name)
    return milvus_conn


def get_registered_person(milvus_conn, person_id: int, output_fields: List[str]) -> dict:
    """
    Get registered data record by person_id.
    Arguments:
        person_id: int = number,
        output_fields: list[str] = ["person_id", "embedding"]
    """
    expr = f'person_id == {person_id}'
    try:
        results = milvus_conn.query(
            expr=expr,
            offset=0,
            limit=10,
            output_fields=output_fields,
            consistency_level="Strong")
        if not results:
            return {"status": "failed",
                    "message": f"person with id: {person_id} not found in milvus database"}
        return {"status": "success",
                "message": f"person with id: {person_id} found in milvus database",
                "person_data": results}
    except MilvusException as excep:
        print(excep)
        return {"status": "failed",
                "message": "error running mivlus database query"}
