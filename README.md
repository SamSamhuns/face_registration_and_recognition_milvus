# Face Registration and Recognition Backend System with uvicron, fastapi, and milvus

<img src="app_docker_compose/app/static/project_flow.png" width="40%" />

[milvus-setup](https://milvus.io/docs/install_standalone-docker.md)

## Setup

### Download model weights

```bash
pip install gdown  # inside virtualenv preferebly
# download model weights
gdown 18dH0l6ESMaHJo3tFMySt0I8LsKcCss3o
unzip models.zip -d app_docker_compose/app/triton_server
rm models.zip
```

## Setup with Docker Compose for Deployment

**Start uvicorn and triton server with a milvus instance for face vector storage & search**

```shell
cd app_docker_compose
# start all services
docker-compose up -d
```

## Setup with Docker for Development

**Allows for rapid prototyping.**

### Build docker

```shell
cd app_docker_compose
bash scripts/build_docker.sh
```

### Local Uvicorn requirements

```bash
cd app_docker_compose
# setup virtual env (conda env is fine as well)
python -m venv venv
source venv/bin/activate
# install all reqs
pip install -r app_docker_compose/requirements.txt
```

### Run servers

#### Start milvus vector database server

```shell
# clear all stopped containers
docker-container prune
# start milvus vector database server with docker-compose
docker-compose up -d etcd minio standalone
# check milvus server status with
docker-compose ps
```

#### Start face model triton=server

```shell
# start triton-server in a docker container exposed onport 8081
docker run -d --rm -p 0.0.0.0:8081:8081 --name uvicorn_trt_server face_recog:latest tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081
# check trtserver status with
docker logs uvicorn_trt_server
```
#### run uvicorn server

```shell
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

### Acknowledgements

-   milvus
-   triton-server
-   uvicorn
-   fastapi
