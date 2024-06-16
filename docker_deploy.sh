#!/bin/bash

TAG=$1

if [ -z $TAG ];
then
  echo "require docker image tag"
  exit;
fi;

echo "Using tag $TAG to deploy"
cat k8s/deployment.yaml | sed "s|<IMAGE>|$TAG|g" > k8s/deployment-output.yaml
kubectl apply -f k8s/deployment-output.yaml
