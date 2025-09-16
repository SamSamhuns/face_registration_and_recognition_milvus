#!/bin/bash
# Start both servers in background

# Run Triton on GRPC port 8081 inside docker. Hardcoded to 8081
tritonserver --model-store app/triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081 --allow-metrics=false --allow-gpu-metrics=false &
P1=$!

# Run FastAPI Server on port 8080. Hardcoded to 8080
python3 app/server.py -p 8080 &
P2=$!
wait $P1 $P2
