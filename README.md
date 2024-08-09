# Person Face Registration and Recognition Backend System with uvicorn, fastapi, milvus, redis and mysql

[![tests](https://github.com/SamSamhuns/face_registration_and_recognition_milvus/actions/workflows/main_test.yml/badge.svg)](https://github.com/SamSamhuns/face_registration_and_recognition_milvus/actions/workflows/main_test.yml)

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

Tested with `docker compose version 1.29.2`.

Backend system for detecting and saving a person's face from images into a vectorized milvus database to run facial recognition on images along with saving the person's data in a redis-cached mysql table for later retrieval. (Note: The system currently only supports one face per image for both face registration and lookup). This repository currently only works with systems with Intel x86_64 cpus and does not support arm64 based systems (i.e. Apple M1 chips).

<img src="app_docker_compose/app/static/project_flow.png" width="60%" alt="project flow" />

- [milvus official setup reference](https://milvus.io/docs/install_standalone-docker.md)

**TABLE OF CONTENTS**

- [Person Face Registration and Recognition Backend System with uvicorn, fastapi, milvus, redis and mysql](#person-face-registration-and-recognition-backend-system-with-uvicorn-fastapi-milvus-redis-and-mysql)
  - [Initial Setup](#initial-setup)
    - [1. Download model weights](#1-download-model-weights)
    - [2. Create .env file](#2-create-env-file)
    - [3. Setup sql schema for storing person data](#3-setup-sql-schema-for-storing-person-data)
    - [4. Create a volume directory to hold user images](#4-create-a-volume-directory-to-hold-user-images)
      - [Notes on editing docker-compose.yml](#notes-on-editing-docker-composeyml)
  - [Setup with Docker Compose for Deployment](#setup-with-docker-compose-for-deployment)
    - [Docker Compose Setup](#docker-compose-setup)
    - [Start uvicorn and triton server with a milvus instance for face vector storage \& search](#start-uvicorn-and-triton-server-with-a-milvus-instance-for-face-vector-storage--search)
  - [Setup with Docker and local python envs for Development](#setup-with-docker-and-local-python-envs-for-development)
    - [1. Build uvicorn\_trt\_server docker](#1-build-uvicorn_trt_server-docker)
    - [2. Local uvicorn requirements](#2-local-uvicorn-requirements)
    - [3. Run servers](#3-run-servers)
      - [3a. Start all required microservices with docker compose](#3a-start-all-required-microservices-with-docker-compose)
      - [3b. Start face model triton-server](#3b-start-face-model-triton-server)
      - [3c. Run fastapi + uvicorn server](#3c-run-fastapi--uvicorn-server)
  - [Running tests](#running-tests)
  - [References](#references)
    - [Encryption of faces](#encryption-of-faces)
    - [Attacks on facial recognition systems](#attacks-on-facial-recognition-systems)
    - [Countermeasures against face recognition attacks](#countermeasures-against-face-recognition-attacks)
    - [Notes on docker compose yml setup](#notes-on-docker-compose-yml-setup)
    - [Notes on triton-server](#notes-on-triton-server)
  - [Acknowledgements](#acknowledgements)

## Initial Setup

### 1. Download model weights

```bash
python3 -m venv venv
source venv/bin/activate
# inside venv/virtualenv/conda
pip install gdown
# download model weights
gdown 12zPEd0IgrEDJU3jcMj5EZWU0Yt7GKTRp
unzip models.zip -d app_docker_compose/app/triton_server
rm models.zip
```

### 2. Create .env file

Create a `.env` file inside `app_docker_compose` based on the following parameters with necessary variables replaced:

```yaml
# download paths
DOWNLOAD_CACHE_PATH="app/.data"
DOWNLOAD_IMAGE_PATH="volumes/person_images"
# http api server
API_SERVER_PORT=8080
# milvus
MILVUS_HOST=standalone
MILVUS_PORT=19530
ATTU_PORT=3000
# mysql mariadb
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=user
MYSQL_PASSWORD=pass
MYSQL_DATABASE=default
MYSQL_PERSON_TABLE=person
MYSQL_ROOT_PASSWORD=admin
# phpmyadmin mariadb
PMA_GUI_PORT=8001
PMA_HOST=${MYSQL_HOST}
PMA_PORT=${MYSQL_PORT}
PMA_USER=${MYSQL_USER}
PMA_PASSWORD=${MYSQL_PASSWORD}
# redis
REDIS_HOST=redis-server
REDIS_PORT=6379
```

Note: Only `.env` allows docker compose to access variables inside `.env` file during build-time. Using `env_file` or the `environment` parameters inside the docker compose file only allows variable access inside containers and not during build time.

### 3. Setup sql schema for storing person data

Schema for creating person data table and the table name should be modified at: `app_docker_compose/app/static/sql/init.sql`

### 4. Create a volume directory to hold user images

```shell
cd app_docker_compose
mkdir -p volumes/person_images
```

#### Notes on editing docker-compose.yml

When changing settings in `docker-compose.yml` for the different services i.e. mysql dbs creation, the existing docker and shared volumes might have to be purged.
To avoid purges, manual creation/edit/deletion of databases must be done with mysql.

<p style="color:red;">WARNING: This will delete all existing users, face-images, and vector records.</p> 

```shell
# run inside the same directory as docker compose.yml
docker compose down
docker volume rm $(docker volume ls -q)
rm -rf volumes
```

## Setup with Docker Compose for Deployment

### Docker Compose Setup

[Uninstall unofficial docker compose conflicting packages](https://docs.docker.com/engine/install/ubuntu/#uninstall-old-versions)

Install `docker compose` from the [official docker site](https://docs.docker.com/compose/install/)

### Start uvicorn and triton server with a milvus instance for face vector storage & search

If there are GPG key errors during the build of `uvicorn_trt_server:latest` image, update docker to the latest version or check out this [nvidia blog for updating cuda linux gpg keys](https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/)

```shell
cd app_docker_compose
# create shared volume directory to store imgs if not already created
mkdir -p volumes/person_images
# build all required containers
docker compose build
# start all services
docker compose up -d
```

Face registration and recognition fastapi will be available at <http://localhost:8080>. The exposed port can be changed with the `API_SERVER_PORT` env variable.

## Setup with Docker and local python envs for Development

**Allows for rapid prototyping.**

Change into main working directory where all subsequent commands must be run.

```shell
cd app_docker_compose
```

### 1. Build uvicorn_trt_server docker

```shell
bash scripts/build_docker.sh
```

### 2. Local uvicorn requirements

To properly resolve host-names in `.env`, the container service names in `docker compose.yml` following must be added to `/etc/hosts` in the local system. This is not required when the fastapi-server is running inside a docker container.

```shell
127.0.0.1  standalone
127.0.0.1  mysql
127.0.0.1  redis-server
```

```bash
# setup virtual env (conda env is fine as well)
python -m venv venv
source venv/bin/activate
# install all reqs
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run servers

#### 3a. Start all required microservices with docker compose

```shell
# clear all stopped containers
docker container prune
# start milvus vector database server with docker compose
docker compose up -d etcd minio standalone attu mysql mysql-admin redis-server
# check milvus server status with
docker compose ps
```

#### 3b. Start face model triton-server

```shell
# start triton-server in a docker container exposed onport 8081
docker run -d --rm -p 127.0.0.1:8081:8081 --name uvicorn_trt_server_cont uvicorn_trt_server:latest tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081
# check trtserver status with
docker logs uvicorn_trt_server_cont
```

#### 3c. Run fastapi + uvicorn server

```shell
python3 app/server.py -p EXPOSED_HTTP_PORT
```

Face registration and recognition fastapi will be available at <http://localhost:EXPOSED_HTTP_PORT>.

## Running tests

```shell
cd app_docker_compose
mkdir -p volumes/person_images
pip install -r requirements.txt
pip install -r tests/requirements.txt
# set up all microservices
docker compose up -d etcd minio standalone attu mysql mysql-admin redis-server
# start face model triton server
docker run -d --rm -p 127.0.0.1:8081:8081 --name uvicorn_trt_server_cont uvicorn_trt_server:latest tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081
# run tests
pytest tests
```

Generating coverage reports

```shell
coverage run -m pytest tests/
coverage report -m -i
```

## References

### Encryption of faces

The [Pyfhel](https://pyfhel.readthedocs.io/en/latest/index.html) library supports FHE schemes with BGV/BFV/CKKS to homomorphically encrypt face vectors so that operations like addition, multiplication, exponentiation or scalar product can be run on the encrypted vectors so that the results are the same if the operations were run on non-encrypted vectors. 

This allows the vectors to be stored in a zero-trust database or untrusted vendor given that only the client has the private key to decrypt the vectors but the server can still run arithmetic operations on the vectors without compromising their security. [See faces can be generated from reversing the face-embeddings from facenet](https://edwardv.com/cs230_project.pdf).

An example script with client-server setup for finding closest embeddings with KNN is provided at [app_docker_compose/scripts/homomorphic_emb_face_search_knn.py].

### Attacks on facial recognition systems

Presentation Attacks (Performed in the physical domain while the faces are presented to the system)

-   2D Spoofing (Printout of the victim's face on paper, Using a mobile device or screen with the victims face, video of the victims face on a screen)
-   3D Spoofing (A crafted 3D face mask)

Indirect Attacks (Performed in the database level after the face image have already been ingested into the digital domain. Standard cybersecurity measures can counter these attacks)

An example script with a train-test setup for training and testing the detection of real-vs-spoofed faces is provided at  [app_docker_compose/scripts/train_spoofed_face_vector_clsf.py].

### Countermeasures against face recognition attacks

-   Stereo camera depth amp reading / 3D face structure reading / 3D face landmark detection
-   Liveliness detection (eye blink detection)
-   Face movement detection & challenge response: nod, smile, head rotation
-   Contextual information techniques (looking for hand)
-   Algorithms:
   -   Texture analysis: Detect artifacts caused by imaging the screen (Moiré patterns) or Local Binary Patterns (LBPs)
   -   Specular feature projections: Train SVM models on specular feature space projections of genuine and spoofed face images for impersonation detection
   -   Frequency analysis: Examine the Fourier domain of the face
   -   Optical flow algorithms: Examine the differences & properties of optical flow generated from 3D objects & 2D planes.
   -   Image quality assessment: Detect spoofs with an ensemble of image quality measures
   -   Depth feature fusion: Deep feature fusion network structure with CNNs & SENet using facial image color features
   -   DNNs: Face classifiers trained on large dataset of real & spoofed images 

Datasets for real vs fake face classification

-   NUAA Photograph Imposter Database
-   [Small set of real & fake face images](https://github.com/SkyThonk/real-and-fake-face-detection)
-   [Kaggle Real and Fake Face Detection Data](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
-   [Deep Face Detection](https://github.com/selimsef/dfdc_deepfake_challenge)

### Notes on docker compose yml setup

Note if services other than the uvicorn web-api are to be exposed such as the milvus or minio servers, alter the `expose` options to published `ports` for access outside the docker containers. When ports are exposed to all interfaces i.e. 0.0.0.0, using `ports` alone is enough to expose the inner port inside the container (`9002` below) to other containers in the same network.

```yaml
expose:
  - "9001"

ports:
  - "9001:9002"
```

For `docker compose version 1.29.2` and `yaml version 3.9`, `mem_limit` can be used with `docker compose up`:

```yaml
mem_limit: 512m
```

For `docker compose version <1.29.2` and `yaml version <3.9`, the following deploy setup can be used with `docker compose --compatibility up`:

```yaml
deploy:
  resources:
    limits:
      memory: 512m
```

### Notes on triton-server

Check saved.model inputs/outputs with `$ saved_model_cli show --dir savemodel_dir --all` after installing tensorflow.

Options for CPU and GPU based models for tritonserver:

```yaml
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
```

## Acknowledgements

-   [milvus](https://milvus.io/)
-   [triton-server](https://developer.nvidia.com/nvidia-triton-inference-server)
-   [mariadb-mysql](https://mariadb.org/)
-   [redis](https://redis.io/)
-   [uvicorn](https://www.uvicorn.org/)
-   [fastapi](https://fastapi.tiangolo.com/)
-   [Pyfhel](https://pyfhel.readthedocs.io/en/latest/index.html)
-   [Spoofing facial recognition](https://towardsdatascience.com/facial-recognition-types-of-attacks-and-anti-spoofing-techniques-9d732080f91e)
