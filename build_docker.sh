#!/bin/bash

echo "Building VORTEX Docker image..."
docker build -t vortex .

echo ""
echo "Docker image 'vortex' built successfully."
echo "You can now run it using ./docker_run.sh"