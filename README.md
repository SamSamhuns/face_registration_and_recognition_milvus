# Face Registration and Recognition System with ElasticSearch


## Setup

### Download model weights

```bash
pip install gdown  # inside virtualenv preferebly
# download model weights
gdown 18dH0l6ESMaHJo3tFMySt0I8LsKcCss3o
unzip models.zip -d app_docker_compose/src/app/triton_server
rm models.zip
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

## Local Uvicorn Server Setup

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
# start triton-server in a docker container exposed onport 8081
docker run --rm -p 0.0.0.0:8081:8081 --name triton_server_cont face_recog tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081
# run uvicorn server
python3 app/server.py -p EXPOSED_HTTP_PORT
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
