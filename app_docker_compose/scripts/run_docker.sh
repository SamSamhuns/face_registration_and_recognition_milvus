#!/bin/bash

def_cont_name=uvicorn_trt_server_cont

helpFunction()
{
   echo ""
   echo "Usage: $0 -p port"
   echo -e "\t-p http_port"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$port" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    echo "Stopping & removing docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    docker rm "$def_cont_name"
    echo "Stopped container '$def_cont_name'"
fi

mkdir -p volumes/person_images  # create shr vol with correct perms
docker run \
      -d \
      -p "0.0.0.0:$port:8080" \
      -v "$PWD/volumes/person_images:/home/triton-server/src/app/person_images" \
      --name "$def_cont_name" \
      uvicorn_trt_server
