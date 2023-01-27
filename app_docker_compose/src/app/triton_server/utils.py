import os
import traceback
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


MAX_FRAMES_TO_GET_FROM_VIDEO = 30


class FlagConfig:
    """stores configurations for prediction"""

    def __init__(self):
        pass


def resize_maintaining_aspect(img, width, height):
    """
    If width and height are both None, no resize is done
    If either width or height is None, resize maintaining aspect
    """
    old_h, old_w, _ = img.shape

    if width is not None and height is not None:
        new_w, new_h = width, height
    elif width is None and height is not None:
        new_h = height
        new_w = (old_w * new_h) // old_h
    elif width is not None and height is None:
        new_w = width
        new_h = (new_w * old_h) // old_w
    else:
        # no resizing done if both width and height are None
        return img
    img = cv2.resize(img, (new_w, new_h))
    return img


def get_client_and_model_metadata_config(FLAGS):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as excep:
        print(f"client creation failed: {excep}" )
        return -1

    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as excep:
        print(f"failed to retrieve the metadata:{excep}")
        return -1

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as excep:
        print(f"failed to retrieve the config: {excep}")
        return -1

    return triton_client, model_metadata, model_config


def requestGenerator(input_data_list, input_name_list, output_name_list, input_dtype_list, FLAGS):

    # set inputs and outputs
    inputs = []
    for i in range(len(input_name_list)):
        inputs.append(grpcclient.InferInput(
            input_name_list[i], input_data_list[i].shape, input_dtype_list[i]))
        inputs[i].set_data_from_numpy(input_data_list[i])

    outputs = []
    for i in range(len(output_name_list)):
        outputs.append(grpcclient.InferRequestedOutput(
            output_name_list[i], class_count=FLAGS.classes))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def parse_model_grpc(model_metadata, model_config):

    input_format_list = []
    input_datatype_list = []
    input_metadata_name_list = []
    for i in range(len(model_metadata.inputs)):
        input_format_list.append(model_config.inputs[i].format)
        input_datatype_list.append(model_metadata.inputs[i].datatype)
        input_metadata_name_list.append(model_metadata.inputs[i].name)
    output_metadata_name_list = []
    for i in range(len(model_metadata.outputs)):
        output_metadata_name_list.append(model_metadata.outputs[i].name)
    # the first input must always be the image array
    s1 = model_metadata.inputs[0].shape[1]
    s2 = model_metadata.inputs[0].shape[2]
    s3 = model_metadata.inputs[0].shape[3]
    return (model_config.max_batch_size, input_metadata_name_list,
            output_metadata_name_list, s1, s2, s3, input_format_list,
            input_datatype_list)


def extract_data_from_media(FLAGS, preprocess_func, media_filenames):
    image_data = []
    all_req_imgs_orig = []
    all_req_imgs_orig_size = []
    fps = None
    for filename in media_filenames:
        if FLAGS.inference_mode == "image":
            try:
                # if an image path is provided instead of a numpy H,W,C image
                if isinstance(filename, str) and os.path.isfile(filename):
                    img = cv2.imread(filename)
                else:
                    img = np.asarray(Image.open(BytesIO(filename)))
                image_data.append(preprocess_func(img=img))
                all_req_imgs_orig_size.append(img.shape)
                if FLAGS.result_save_dir is not None:
                    all_req_imgs_orig.append(img)
                fps = 1
            except Exception as excep:
                traceback.print_exc()
                print(f"{excep}. Failed to process image {filename}")
        elif FLAGS.inference_mode == "video":
            try:
                cap = cv2.VideoCapture(filename)
                fps = round(cap.get(cv2.CAP_PROP_FPS))  # take 1 frame every fps frames

                # check num of channels
                ret, frame = cap.read()
                orig_shape = frame.shape
                if ret and frame.shape[-1] != 3:
                    raise Exception("Video must have 3 channels")

                # set opencv reader to vid start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                orig_vid, vid = [], []
                i = 0
                while cap.isOpened() and len(vid) <= MAX_FRAMES_TO_GET_FROM_VIDEO:
                    ret, frame = cap.read()
                    if ret:
                        if i % fps == 0:
                            if FLAGS.result_save_dir is not None:
                                orig_vid.append(np.copy(frame))
                            vid.append(preprocess_func(frame))
                    else:
                        break
                    i += 1
                image_data = vid
                all_req_imgs_orig = orig_vid
                all_req_imgs_orig_size = np.array(
                    [len(image_data), *orig_shape])
                cap.release()
            except Exception as excep:
                traceback.print_exc()
                print(f"{excep}. Failed to process video {filename}")
    return image_data, all_req_imgs_orig, all_req_imgs_orig_size


def get_inference_responses(image_data_list, FLAGS, trt_inf_data):
    triton_client, input_name, output_name, input_dtype, max_batch_size = trt_inf_data
    responses = []
    image_idx = 0
    last_request = False
    sent_count = 0

    image_data = image_data_list[0]
    while not last_request:
        repeated_image_data = []

        for idx in range(FLAGS.batch_size):
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True
        if max_batch_size > 0:
            batched_image_data = np.stack(
                repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]
        if max_batch_size == 0:
            batched_image_data = np.expand_dims(batched_image_data, 0)

        input_image_data = [batched_image_data]
        # if more inputs are present, i.e. for edetlite4_modified
        # then add other inputs to input_image_data
        if len(image_data_list) > 1:
            for in_data in image_data_list[1:]:
                input_image_data.append(in_data)
        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    input_image_data, input_name, output_name, input_dtype, FLAGS):
                sent_count += 1
                responses.append(
                    triton_client.infer(FLAGS.model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=FLAGS.model_version,
                                        outputs=outputs))

        except InferenceServerException as excep:
            traceback.print_exc()
            print(f"inference failed: {excep}")
            return -1

    return responses
