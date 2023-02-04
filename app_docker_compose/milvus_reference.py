"""
MIlvus face recognition example
https://github.com/milvus-io/bootcamp/blob/master/solutions/image/face_recognition_system/face_recognition_bootcamp.ipynb
"""
import random
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility


# create connec to milvus server
connections.connect(
    alias="default",
    host='127.0.0.1',
    port='19530'
)

# CREATE SCHEMA FOR A COLLECTION
# The collection to create must contain a primary key field and a vector field. INT64 and String are supported data type on primary key field.
# max number of fields in collection: 64
book_id = FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True)
book_name = FieldSchema(name="book_name", dtype=DataType.VARCHAR, max_length=200)
word_count = FieldSchema(name="word_count", dtype=DataType.INT64)
# maximum vector dim length: 32,768
book_intro = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=2)
schema = CollectionSchema(fields=[book_id, book_name, word_count, book_intro], description="Test book search")

collection_name = "book"

if not utility.has_collection(collection_name):
    # create a collection which consists of one or more partitions
    collection = Collection(
        name=collection_name,  # max size 255 chars
        schema=schema,
        using='default',
        shards_num=2  # max num of shards in collection: 256
    )
    # Get an existing collection.
    stored_collection = Collection("book")
    print(stored_collection)

    # INSERT

    # random data to insert into milvus collection
    data = [
        [i for i in range(2000)],  # book ids
        [str(i) for i in range(2000)],  # book names
        [i for i in range(10000, 12000)],  # word counts
        [[random.random() for _ in range(2)] for _ in range(2000)],  # book intros
    ]
    # insert data into collection book
    stored_collection.insert(data)
    # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    stored_collection.flush()

    # CREATE VECTOR INDEX FOR EFFICIENT SEARCH

    # Vector indexes are an organizational unit of metadata used to accelerate vector similarity search.
    # Without the index built on vectors, Milvus will perform a brute-force search by default.

    # By default, Milvus does not index a segment with less than 1,024 rows.
    # To change this parameter, configure rootCoord.minSegmentSizeToEnableIndex in milvus.yaml.
    # milvus index: https://milvus.io/docs/index.md
    # milvus similarity metrics: https://milvus.io/docs/metric.md
    # The following example builds a 1024-cluster IVF_FLAT index with Euclidean distance (L2) as the similarity metric.
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

    try:
        stored_collection.release()
    except Exception as e:
        print(e)

    # Build the index by specifying the vector field name and index parameters.
    stored_collection.create_index(
        field_name="book_intro",
        index_params=index_params
    )
else:
    # Get an existing collection.
    stored_collection = Collection("book")

# All search and query operations within Milvus are executed in memory.
# Load the collection to memory before conducting a vector similarity search.
stored_collection.load()

# Prepare search parameters
# Prepare the parameters that suit your search scenario.
# The following example defines that the search will calculate the distance with Euclidean distance,
# and retrieve vectors from ten closest clusters built by the IVF_FLAT index.
search_params = {
    "metric_type": "L2", 
    "params": {"nprobe": 10},
    "offset": 5
}

# milvus consistency: https://milvus.io/docs/consistency.md
results = stored_collection.search(
    data=[[0.1, 0.2]],
    anns_field="book_intro",
    param=search_params,
    limit=10,
    expr=None,
    consistency_level="Strong"
)

print(results[0].ids)
print(results[0].distances)

# SEARCH
expr = f"book_id in [{','.join(map(str, results[0].ids))}]"
print("Search expr: "+expr)
result_data = stored_collection.query(
  expr = expr,
  offset = 0,
  limit = 10, 
  output_fields = ["book_id", "book_name", "word_count", "book_intro"],
  consistency_level="Strong"
)
print(result_data)


# release the connecion from memory
stored_collection.release()

# close all connections
connections.disconnect(
    "default"
)
