#!/bin/bash

pod_name=$(kubectl get pods -n alethic --selector=app=alethic-ism-api -o jsonpath="{.items[0].metadata.name}")
echo "pod name: $pod_name"
kubectl -n alethic port-forward pod/$pod_name 8001:80
