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

echo "Identifying ISM Core and ISM DB libraries"
conda_ism_core_path=$(find $CONDA_PACKAGE_PATH_ISM_CORE -type f -name "alethic-ism-core*.tar.gz")
conda_ism_core_path=$(basename $conda_ism_core_path)
echo "Using Conda ISM Core Library Path: $conda_ism_core_path"

conda_ism_db_path=$(find $CONDA_PACKAGE_PATH_ISM_DB -type f -name "alethic-ism-db*.tar.gz")
conda_ism_db_path=$(basename $conda_ism_db_path)
echo "Using Conda ISM DB Library Path: $conda_ism_db_path"

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
