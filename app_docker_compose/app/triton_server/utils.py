"""
This module contains utility functions for inference with Triton Inference Server.
"""
import os
import traceback
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class FlagConfig:
    """stores configurations for inference"""

    def __init__(self):
        pass


def resize_maintaining_aspect(img, width, height):
    """
    Resizes an image while maintaining aspect ratio.
    If width and height are both None, no resize is done. 
    If either width or height is None, resize maintaining aspect.
    Args:
        img: The input image.
        width: The desired width of the resized image. If None, aspect ratio is maintained.
        height: The desired height of the resized image. If None, aspect ratio is maintained.
    Returns:
        img: resize cv2 image
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
    """
    Creates a Triton Inference Server client and retrieves model metadata and configuration.
    Args:
        FLAGS: The configuration parameters.
    Returns:
        triton_client: The Triton Inference Server client.
        model_metadata: The model metadata.
        model_config: The model configuration.
        -1: If there was an error creating the client or retrieving the metadata/config.
    """
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
    """
    Generates inference requests for the Triton Inference Server.
    Args:
        input_data_list: The list of input data.
        input_name_list: The list of input names.
        output_name_list: The list of output names.
        input_dtype_list: The list of input data types.
        FLAGS: The configuration parameters.
    Yields:
        inputs: The inputs for the inference request.
        outputs: The requested outputs for the inference request.
        model_name: The name of the model.
        model_version: The version of the model.
    """
    # set inputs and outputs
    inputs = []
    for i, input_name in enumerate(input_name_list):
        inputs.append(grpcclient.InferInput(
            input_name, input_data_list[i].shape, input_dtype_list[i]))
        inputs[i].set_data_from_numpy(input_data_list[i])

    outputs = []
    for output_name in output_name_list:
        outputs.append(grpcclient.InferRequestedOutput(
            output_name, class_count=FLAGS.classes))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def parse_model_grpc(model_metadata, model_config):
    """
    Parses the model metadata and configuration to extract necessary information for inference.
    Args:
        model_metadata: The model metadata.
        model_config: The model configuration.
    Returns:
        max_batch_size: The maximum batch size for the model.
        input_name_list: The list of input names.
        output_name_list: The list of output names.
        s1: The height of the input image.
        s2: The width of the input image.
        s3: The number of channels in the input image.
        input_format_list: The list of input formats.
        input_datatype_list: The list of input data types.
    """
    input_format_list = []
    input_datatype_list = []
    input_metadata_name_list = []
    for i in range(len(model_metadata.inputs)):
        input_format_list.append(model_config.input[i].format)
        input_datatype_list.append(model_metadata.inputs[i].datatype)
        input_metadata_name_list.append(model_metadata.inputs[i].name)
    output_metadata_name_list = []
    for i in range(len(model_metadata.outputs)):
        output_metadata_name_list.append(model_metadata.outputs[i].name)
    # the first input must always be the image array
    s_1 = model_metadata.inputs[0].shape[1]
    s_2 = model_metadata.inputs[0].shape[2]
    s_3 = model_metadata.inputs[0].shape[3]
    return (model_config.max_batch_size, input_metadata_name_list,
            output_metadata_name_list, s_1, s_2, s_3, input_format_list,
            input_datatype_list)


def extract_data_from_media(FLAGS, preprocess_func, media_filenames):
    """
    Extracts input data from media files for inference.
    Args:
        FLAGS: The configuration parameters.
        preprocess_func: The preprocessing function to apply to the input data.
        media_filenames: The list of media filenames.
    Returns:
        image_data: The list of preprocessed input data.
        all_req_imgs_orig: The list of original input images.
        all_req_imgs_orig_size: The list of sizes of the original input images.
    """
    image_data = []
    all_req_imgs_orig = []
    all_req_imgs_orig_size = []

    for filename in media_filenames:
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
        except Exception as excep:
            traceback.print_exc()
            print(f"{excep}. Failed to process image {filename}")

    return image_data, all_req_imgs_orig, all_req_imgs_orig_size


def get_inference_responses(image_data_list, FLAGS, trt_inf_data):
    """
    Performs inference using the Triton Inference Server and returns the responses.
    Args:
        image_data_list: The list of input data.
        FLAGS: The configuration parameters.
        trt_inf_data: The Triton Inference Server data containing:
            triton_client: The Triton Inference Server client.
            input_name: The list of input names.
            output_name: The list of output names.
            input_dtype: The list of input data types.
            max_batch_size: The maximum batch size for the model.
    Returns:
        responses: The list of inference responses.
        -1: If there was an error performing inference.
    """
    triton_client, input_name, output_name, input_dtype, max_batch_size = trt_inf_data
    responses = []
    image_idx = 0
    last_request = False
    sent_count = 0

    image_data = image_data_list[0]
    while not last_request:
        repeated_image_data = []

        for _ in range(FLAGS.batch_size):
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
                    triton_client.infer(model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=model_version,
                                        outputs=outputs))

        except InferenceServerException as excep:
            traceback.print_exc()
            print(f"inference failed: {excep}")
            return -1

    return responses
