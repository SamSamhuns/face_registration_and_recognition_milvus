"""
Inference functions
"""
import os
from pathlib import Path

import numpy as np
from triton_server.inference_trtserver import run_inference


def register_face(model_name: str,
                  file_path: str,
                  threshold: float,
                  feat_save_dir: str) -> dict:
    """
    Detects faces in image from the file_path and saves the face feature vector.
    """
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        return_mode="json")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    # save face feature vector to local system for now
    face_vector = pred_dict["face_feats"]
    vector_save_path = os.path.join(feat_save_dir, Path(file_path).stem+".npy")
    np.save(vector_save_path, face_vector)

    # TODO save vectors in a database or ElasticSearch database
    return {"status": "success", "message": "face successfully saved"}


def recognize_face(model_name: str,
                   file_path: str,
                   threshold: float,
                   feat_save_dir: str) -> dict:
    """
    Detects faces in image from the file_path and finds the most similar face vector 
    from a set of saved face vectors
    """
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=threshold,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        return_mode="json")
    print(pred_dict)
    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed", "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    # find and return the closest face
    # TODO this process is currently extremely inefficient, instead a database or eks system should be used
    # a threshold should also be added so that a new completely new face is reported as non-present in the database
    ref_face_vector = pred_dict["face_feats"]
    feat_path_list = [path for path in Path(feat_save_dir).iterdir() if path.suffix == ".npy"]
    feat_list = [np.load(path) for path in feat_path_list]

    # get closest face based on euclidean dist & return
    feat_dist = [np.linalg.norm(face_feat - ref_face_vector) for face_feat in feat_list]
    index_min = min(range(len(feat_dist)), key=feat_dist.__getitem__)
    closest_face = feat_path_list[index_min].stem.split('_')[0]

    return {"status": "success", "message": f"Detected face matches {closest_face}", "match_name": closest_face}
