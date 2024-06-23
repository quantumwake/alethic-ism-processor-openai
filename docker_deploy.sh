#!/bin/bash

# Function to print usage
print_usage() {
  echo "Usage: $0 [-a app_name] [-n docker_namespace] [-t tag]"
  echo "  -a app_name           Application name (default: alethic-ism-core)"
  echo "  -n docker_namespace   Docker namespace (default: krasaee)"
  echo "  -t tag                Docker image tag"
}

# Default values
APP_NAME=$(pwd | sed -e 's/^.*\///g')
DOCKER_NAMESPACE="krasaee"
TAG=""

# Parse command line arguments
while getopts 'a:n:t:' flag; do
  case "${flag}" in
    a) APP_NAME="${OPTARG}" ;;
    n) DOCKER_NAMESPACE="${OPTARG}" ;;
    t) TAG="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# If TAG is not provided, generate it using GIT_COMMIT_ID
if [ -z "$TAG" ]; then
  GIT_COMMIT_ID=$(git rev-parse HEAD)
  TAG="$DOCKER_NAMESPACE/$APP_NAME:$GIT_COMMIT_ID"
fi

echo "Pusing docker image"
docker push $TAG

echo "Using tag $TAG to deploy"
cat k8s/deployment.yaml | sed "s|<IMAGE>|$TAG|g" > k8s/deployment-output.yaml
kubectl apply -f k8s/deployment-output.yaml
