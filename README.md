# Face Registration and Recognition System with ElasticSearch


## Setup

### Download model weights

```bash
pip install gdown  # inside virtualenv preferebly
# download facenet savedmodel weights
gdown 1Ofy7fIv5YrmbvM5BkSnw-E4_uu0VujK7
unzip facenet_trtserver.zip -d app_docker_compose/src/app/triton_server/models
rm facenet_trtserver.zip
```

## Setup with Docker (Recommended)

### Setup requirements

```shell
cd app_docker_compose/src
bash scripts/build_docker.sh
```

### Run servers

```shell
bash scripts/run_docker -p EXPOSED_HTTP_PORT
```

## Local Setup

### Setup requirements

```bash
# setup virtual env (conda env is fine as well)
python -m venv venv
source venv/bin/activate
# install all reqs
pip install -r app_docker_compose/src/requirements.txt
```

### Run servers

```shell
cd app_docker_compose/src
# run uvicorn and triton server
python3 app/server.py -p EXPOSED_HTTP_PORT & tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081 &
```

#### Notes on triton-server

Check saved.model inputs/outputs with `$ saved_model_cli show --dir savemodel_dir --all` after installing tensorflow.

Options for CPU and GPU based models for tritonserver:

    # CPU mode
    instance_group [
        {
          count: 1
          kind: KIND_CPU
        }
      ]

    # GPU mode
    instance_group [
        {
          count: 1
          kind: KIND_GPU
          gpus: [ 0 ]
        }
      ]
