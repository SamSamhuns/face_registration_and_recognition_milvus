#!/bin/bash
mkdir -p volumes/person_images  # create shr vol with correct perms
docker build -t uvicorn_trt_server .
