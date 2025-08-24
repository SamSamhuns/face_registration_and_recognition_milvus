"""
milvus api functions
"""

import logging

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections, utility

logger = logging.getLogger("milvus_api")


def get_milvus_collec_conn(
    collection_name: str,
    milvus_host: str = "0.0.0.0",
    milvus_port: int = 19530,
    vector_dim: int = 128,
    metric_type: str = "L2",
    index_type: str = "IVF_FLAT",
    index_metric_params: dict = None,
):
    """
    Gets the milvus collection connection with the given collection name otherwise creates a new one
    Note: index_metric_params: dict = {"nlist": 4096} for index_type == "IVF_FLAT"
    """
    # connect to milvus
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(
                name="person_id", dtype=DataType.INT64, description="persons unique id", is_primary=True, auto_id=False
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="embedding vectors", dim=vector_dim),
        ]
        schema = CollectionSchema(fields=fields, description="face recognition system")
        milvus_collec_conn = Collection(
            name=collection_name, consistency_level="Strong", schema=schema, using="default"
        )
        logger.info("Collection %s created.✅️", collection_name)

        # Indexing the milvus_collec_conn
        logger.info("Indexing the Collection...🕓")
        # create IVF_FLAT index for milvus_collec_conn.
        index_params = {"metric_type": metric_type, "index_type": index_type, "params": index_metric_params}
        milvus_collec_conn.create_index(field_name="embedding", index_params=index_params)
        logger.info("Collection %s indexed.✅️", collection_name)
    else:
        logger.info("Collection %s present already.✅️", collection_name)
        milvus_collec_conn = Collection(collection_name)
    return milvus_collec_conn


def get_registered_person_milvus(milvus_collec_conn, person_id: int, output_fields: list[str]) -> dict:
    """
    Get registered data record by person_id.
    Arguments:
        person_id: int = number,
        output_fields: list[str] = ["person_id", "embedding"]
    """
    expr = f"person_id == {person_id}"
    try:
        results = milvus_collec_conn.query(
            expr=expr, offset=0, limit=10, output_fields=output_fields, consistency_level="Strong"
        )
        if not results:
            return {"status": "failed", "message": f"person with id: {person_id} not found in milvus database"}
        return {
            "status": "success",
            "message": f"person with id: {person_id} found in milvus database",
            "person_data": results,
        }
    except MilvusException as excep:
        logger.error("%s: error running mivlus database query", excep)
        return {"status": "failed", "message": "error running mivlus database query"}
