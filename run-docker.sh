#!/bin/bash

# General info about new container / image
# This will create a container where the code will be in ./project subfolder
CONTAINER_NAME=kaust/mtl_low
CONTAINER_SHORTCUT_NAME=mtl_low
SUBDIR_NAME=project
PORT_HOST=8883
TAG=latest

# Specs of the current user. These will be arguments to Dockerfile
WORKDIR=$PWD
THIS_UID=`id -u`
THIS_GID=`id -g`
THIS_USER=$USER

echo "Starting $CONTAINER_NAME:$TAG container..."
echo "User: $THIS_USER, UID: $THIS_UID, GID: $THIS_GID"

# Build a container
docker build \
--build-arg MYUID=$THIS_UID \
--build-arg MYGID=$THIS_GID \
--build-arg MYUSER=$THIS_USER \
-t $CONTAINER_NAME .

# Start container. Note that sudo is not necessary, but 
# the user won't have root rights anyway
# --rm \
docker run \
    -it \
    --rm \
    --runtime=nvidia \
    --privileged \
    -v /dev:/dev \
    -p $PORT_HOST:8888 \
    --shm-size 4G \
    --gpus all \
    -v $WORKDIR:/workspace/$SUBDIR_NAME \
    --name $CONTAINER_SHORTCUT_NAME \
    $CONTAINER_NAME:$TAG
