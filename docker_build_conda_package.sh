#!/bin/bash

APP_NAME="alethic-ism-processor-openai"
DOCKER_NAMESPACE="krasaee"
CONDA_PACKAGE_PATH_ISM_CORE="../alethic-ism-core"
CONDA_PACKAGE_PATH_ISM_DB="../alethic-ism-db"
GIT_COMMIT_ID=$(git rev-parse HEAD)
TAG="$DOCKER_NAMESPACE/$APP_NAME:$GIT_COMMIT_ID"

ARCH=$1
if [ -z "$ARCH" ];
then
  ARCH="linux/amd64"
  #TODO check operating system ARCH="linux/arm64"
fi;

echo "Using arch: $ARCH for image tag $TAG"

echo "Identifying ISM core library"
conda_ism_core_path=$(ls -ltr $CONDA_PACKAGE_PATH_ISM_CORE/alethic-ism-core*.tar.gz | awk '{print $9}' | tail -n 1)
conda_ism_core_path=$(basename $conda_ism_core_path)
echo "Using Conda ISM core library: $conda_ism_core_path"

echo "Identifying ISM database library"
conda_ism_db_path=$(ls -ltr $CONDA_PACKAGE_PATH_ISM_DB/alethic-ism-db*.tar.gz | awk '{print $9}' | tail -n 1)
conda_ism_db_path=$(basename $conda_ism_db_path)
echo "Using Conda ISM database library: $conda_ism_db_path"

if [ ! -z $conda_ism_db_path ] && [ ! -z $conda_ism_core_path ];
then
  echo "Copying ISM core and db packages to local for docker build to consume"
  cp $CONDA_PACKAGE_PATH_ISM_CORE/$conda_ism_core_path $conda_ism_core_path
  cp $CONDA_PACKAGE_PATH_ISM_DB/$conda_ism_db_path $conda_ism_db_path

  echo "Starting docker build for $TAG"
  docker build \
   --platform $ARCH \
   --progress=plain -t $TAG \
   --build-arg CONDA_ISM_CORE_PATH=$conda_ism_core_path \
   --build-arg CONDA_ISM_DB_PATH=$conda_ism_db_path \
   --no-cache .

  docker push $TAG
  echo "Use $TAG to deploy to kubernetes"
fi;
