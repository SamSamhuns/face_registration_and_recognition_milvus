"""
Inference with tritonserver. The tritonserver must be runnign the background
"""
from functools import partial
from pathlib import Path
from typing import Tuple
import time
import os

import numpy as np
import cv2

from triton_server.utils import FlagConfig
from triton_server.utils import extract_data_from_media, get_client_and_model_metadata_config
from triton_server.utils import parse_model_grpc, get_inference_responses
from utils.image import pad_resize_image, scale_coords, draw_bbox_on_image


FLAGS = FlagConfig()


def preprocess(img: np.ndarray, width: int = 448, height: int = 448, new_type: type = np.float32) -> np.ndarray:
    """
    Preprocess input cv2 image
    Args:
        img: cv2 input image
        width: model input width
        height: model input height
        new_type: type to which the image is converted to
    """
    width = 448 if width is None else width
    height = 448 if height is None else height
    resized = pad_resize_image(img, (width, height))
    img_in = np.transpose(resized, (2, 0, 1)).astype(new_type)  # HWC -> CHW
    return img_in


def postprocess(results, orig_img_size: Tuple[int, int], input_img_size: Tuple[int, int]) -> dict:
    """
    Postprocess detections & face features from triton-server
    Args:
        results: triton server inference result
        orig_img_size: (width, height)
        input_img_size: (width, height)
    """
    predictions = {"face_feats": [], "face_detections": []}

    results_arr = results.as_numpy("ENSEMBLE_FACE_FEAT")
    if results_arr.any():
        for i, result in enumerate(results_arr):
            if FLAGS.face_feat_model[0].decode() == "facenet_trtserver":
                pass   # result = result
            elif FLAGS.face_feat_model[0].decode() == "face-reidentification-retail-0095":
                result = result.squeeze()
            predictions["face_feats"].append(result)

    box_arr = results.as_numpy("ENSEMBLE_FACE_DETECTOR_BBOXES").copy()
    conf_arr = results.as_numpy("ENSEMBLE_FACE_DETECTOR_CONFS").tolist()

    if box_arr is not None and box_arr.any():
        model_w, model_h = input_img_size
        orig_w, orig_h = orig_img_size
        box_arr *= [model_w, model_h, model_w, model_h]
        box_arr = scale_coords((model_h, model_w), box_arr, (orig_h, orig_w))
        for i, result in enumerate(box_arr):
            result = result.copy()
            x_min, y_min, x_max, y_max = result.astype(int).tolist()
            conf = conf_arr[i]
            predictions["face_detections"].append(
                {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "confidence": conf})
    return predictions


def run_inference(media_filename: str,
                  face_feat_model: str = "facenet_trtserver",
                  face_det_thres: float = 0.55,
                  face_bbox_area_thres: float = 0.10,
                  face_count_thres: int = 1,
                  save_result_dir: str = None,  # set to None prevent saving
                  debug: bool = False,
                  port: int = 8081,
                  return_mode: str = "json") -> dict:
    """
    Note: only one image is assumed for inferencing
    Returns a dict with params status and message along with optionally other parameters
    status != 0 means failure
    """
    FLAGS.media_filename = media_filename
    FLAGS.face_feat_model = np.array([face_feat_model.encode()])
    FLAGS.face_det_thres = face_det_thres
    FLAGS.face_bbox_area_thres = face_bbox_area_thres
    FLAGS.face_count_thres = face_count_thres
    FLAGS.result_save_dir = save_result_dir
    FLAGS.model_version = ""  # empty str means use latest
    FLAGS.protocol = "grpc"
    FLAGS.url = f"127.0.0.1:{port}"
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.debug = debug
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = None
    FLAGS.fixed_input_height = None
    if face_feat_model == "facenet_trtserver":
        FLAGS.model_name = "ensemble_face_facenet"
    elif face_feat_model == "face-reidentification-retail-0095":
        FLAGS.model_name = "ensemble_face_face_reid"
    else:
        raise NotImplementedError(
            f"face_feat_model {face_feat_model} is not implemented")
    start_time = time.time()

    if FLAGS.result_save_dir is not None:
        FLAGS.result_save_dir = os.path.join(
            save_result_dir, f"{FLAGS.model_name}")
        os.makedirs(FLAGS.result_save_dir, exist_ok=True)
    if FLAGS.debug:
        print(f"Running model {FLAGS.model_name}")

    model_info = get_client_and_model_metadata_config(FLAGS)
    if model_info == -1:  # error getting model info
        err_msg = "Error getting model info"
        print(err_msg)
        return {"status": -1, "message": err_msg}
    triton_client, model_metadata, model_config = model_info

    # input_name, output_name, format, dtype are all lists
    # parse returns: max_batch_size, input_name, output_name, model_c, model_h, model_w, format, dtype
    max_batch_size, input_name, output_name, _, model_h, model_w, _, dtype = parse_model_grpc(
        model_metadata, model_config.config)

    # check for dynamic input shapes
    if model_h == -1:
        model_h = FLAGS.fixed_input_height
    if model_w == -1:
        model_w = FLAGS.fixed_input_width

    nptype_dict = {"UINT8": np.uint8, "FP32": np.float32, "FP16": np.float16}
    # Important, make sure the first input is the input image
    image_input_idx = 0
    preprocess_dtype = partial(
        preprocess, new_type=nptype_dict[dtype[image_input_idx]])
    # all_reqested_images_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_reqested_images_orig, all_req_imgs_orig_size = extract_data_from_media(
        FLAGS, preprocess_dtype, [FLAGS.media_filename])
    if len(image_data) == 0:
        err_msg = "Image data is missing. Aborting inference"
        print(err_msg)
        return {"status": -2, "message": err_msg}

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    image_data_list = [image_data,
                       FLAGS.face_feat_model,
                       np.array([FLAGS.face_det_thres], dtype=np.float32),
                       np.array([FLAGS.face_bbox_area_thres],
                                dtype=np.float32),
                       np.array([FLAGS.face_count_thres], dtype=np.int32)]
    # get inference results
    responses = get_inference_responses(
        image_data_list, FLAGS, trt_inf_data)

    if responses == -1:
        err_msg = f"""Detected of faces detected might have exceeded allowed number of faces ({face_count_thres}). Or there might be other issues during inference. Check server logs"""
        print(err_msg)
        return {"status": -3, "message": err_msg}

    orig_h, orig_w = all_req_imgs_orig_size[0][:2]
    final_result = postprocess(
        responses[0], (orig_w, orig_h), (model_w, model_h))
    if debug:
        print(f"Triton-server inference time {time.time() - start_time:.2f}s")
    if return_mode == "json":
        return {"status": 0, "message": "inference_complete", **final_result}
    if return_mode == "image":
        orig_cv2_img = all_reqested_images_orig[0]
        labels, confs, boxes = [], [], []
        for box in final_result["face_detections"]:
            x_min, y_min, x_max, y_max = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            labels.append("face")
            confs.append(box["confidence"])
            boxes.append(np.asarray([x_min, y_min, x_max, y_max]))

        draw_bbox_on_image(orig_cv2_img, boxes, confs, labels)
        if FLAGS.result_save_dir is not None:
            save_path = os.path.join(FLAGS.result_save_dir, Path(
                FLAGS.media_filename).stem+".jpg")
            cv2.imwrite(save_path, orig_cv2_img)
        return {"status": 0, "message": "inference_complete", "image": orig_cv2_img, **final_result}

    err_msg = f"'{return_mode}' Return mode not supported"
    print(err_msg)
    return {"status": -4, "message": err_msg}


def main():
    """
    Demo inference run
    """
    run_inference(
        "static/faces/one_face_1.jpg",
        face_feat_model="facenet_trtserver",
        face_det_thres=0.55,
        face_bbox_area_thres=0.10,
        face_count_thres=1,
        save_result_dir="output",  # set to None prevent saving
        debug=True,
        port=8081,
        return_mode="image")


if __name__ == "__main__":
    main()
