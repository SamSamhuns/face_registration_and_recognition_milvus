"""

Insert faces and related person information into the milvus and sql database

CelebA dataset website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Aligned faces dataset: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    images: img/img_align_celeba.zip 
    annotations: Anno 

Note only one face per celebrity is inserted into database (i.e. the first face alphabetically inside img_align_celeba)

Faces and related data must be inserted directly into milvus and mysql to reduce latency
First extract embeddings from faces and insert into milvus
Second insert related face data into mysql database

pip install towhee==1.0.0rc1
"""
import os.path as osp
import glob
import random
from datetime import date

import uuid
import pymysql
from pymysql.cursors import DictCursor

from app.api.mysql import insert_person_data_into_sql
from app.api.milvus import get_milvus_collec_conn
from app.models.model import PersonModel, ModelType
import app.config as cfg


IMG_EXTS = set([".jpg", ".png", ".jpeg"])

# Connect to MySQL
mysql_conn = pymysql.connect(
    host=cfg.MYSQL_HOST,
    port=cfg.MYSQL_PORT,
    user=cfg.MYSQL_USER,
    password=cfg.MYSQL_PASSWORD,
    db=cfg.MYSQL_DATABASE,
    cursorclass=DictCursor)

# connect to milvus connec
milvus_collec_conn = get_milvus_collec_conn(
    collection_name=cfg.FACE_COLLECTION_NAME,
    milvus_host=cfg.MILVUS_HOST,
    milvus_port=cfg.MILVUS_PORT,
    vector_dim=cfg.FACE_VECTOR_DIM,
    metric_type=cfg.FACE_METRIC_TYPE,
    index_type=cfg.FACE_INDEX_TYPE,
    index_metric_params={"nlist": cfg.FACE_INDEX_NLIST})


def insert_embeddings_into_milvus_towhee(img_dir: str):
    """
    reference nb: https://github.com/towhee-io/examples/blob/main/image/reverse_image_search/1_build_image_search_engine.ipynb
    """
    from towhee import pipe, ops, DataCollection, register
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

    insert_src_pat = osp.join(img_dir, "*.jpg")

    def load_image(path_pattern):
        """Yield image paths"""
        for item in glob.glob(path_pattern):
            yield item

    @register
    def gen_int_id(x = None):
        """Get a unique int uuid"""
        return uuid.uuid1().int >> 64

    @register
    def get_facenet_emb(vec):
        """Get the embedding vector from a facenet model output"""
        return [list(map(float, vec[0]["embedding"]))]

    # Face embedding pipeline
    p_embed = (
        pipe.input('src')
        .flat_map('src', 'img_path', load_image)
        .map('img_path', 'img_id', gen_int_id())
        .map('img_path', 'img', ops.image_decode())
        .map('img', 'vec', ops.face_embedding.deepface(model_name='Facenet'))
        .map('vec', 'emb', get_facenet_emb())
    )

    # WARNING, the processing time could be huge
    # p_out = p_embed.output("emb")(insert_src_pat).get()
    # print(len(p_out))

    # Insert pipeline
    p_insert = (
        p_embed.map(('emb', 'img_id'), 'mr', ops.ann_insert.milvus_client(
                    host=cfg.MILVUS_HOST,
                    port=cfg.MILVUS_PORT,
                    collection_name=cfg.FACE_COLLECTION_NAME
                    ))
        .output('mr')
    )

    # Insert data
    p_insert(insert_src_pat)

    # Check collection
    print('Number of data inserted:', milvus_collec_conn.num_entities)


def insert_embeddings_into_milvus_trt_sever(img_dir: str):
    """
    Inserts embeddings data in bulk into milvus
    """
    import sys
    sys.path.append("app")
    from app.triton_server.inference_trtserver import run_inference

    imgs = sorted(glob.glob(osp.join(img_dir, "*")))
    imgs = [ip for ip in imgs if osp.splitext(ip)[-1] in IMG_EXTS]

    for file_path in imgs:
        pred_dict = run_inference(
            file_path,
            face_feat_model=ModelType.SLOW.value,
            face_det_thres=0.3,
            face_bbox_area_thres=0.10,
            face_count_thres=1,
            return_mode="json")

        person_id = ""
        # insert face_vector into milvus milvus_collec_conn
        face_vector = pred_dict["face_feats"][0].tolist()
        data = [[face_vector], [person_id]]
        milvus_collec_conn.insert(data)


def insert_data_into_mysql(img_dir: str):
    """
    Inserts data in bulk into mysql
    Each data should match the data model in PersonModel app/models/model.py
    """
    imgs = sorted(glob.glob(osp.join(img_dir, "*")))
    imgs = [ip for ip in imgs if osp.splitext(ip)[-1] in IMG_EXTS]
    imgs_iids = list(range(len(imgs)))
    data_list = [PersonModel(
        ID=iid,
        name=f"person_{iid}",
        birthdate=date(1971, random.randint(1, 12), random.randint(1, 28)),
        country=f"country_{random.randint(1, 1000)}").dict()
        for iid in imgs_iids]

    ins = 0
    for data in data_list:
        rtn = insert_person_data_into_sql(
            mysql_conn, cfg.MYSQL_CUR_TABLE, data)
        if rtn["status"] == "success":
            ins += 1
    print(f"{ins} records successfully inserted into mysql table out of {len(data_list)} original data")


def main():
    """
    Main function
    """
    insert_embeddings_into_milvus_towhee(
        "/home/mluser/sam/face_registration_and_recognition_milvus/app_docker_compose/volumes/img_align_celeba_unique")
    # insert_data_into_mysql(
    #     "/home/mluser/sam/face_registration_and_recognition_milvus/app_docker_compose/volumes/img_align_celeba_unique")


if __name__ == "__main__":
    main()
