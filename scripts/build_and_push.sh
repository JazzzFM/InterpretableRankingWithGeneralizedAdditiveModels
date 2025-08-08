#!/usr/bin/env bash
set -euo pipefail

: "${ACR_NAME:?}"

ACR_LOGIN_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)
echo "Login to ACR: $ACR_LOGIN_SERVER"
az acr login -n "$ACR_NAME"

echo "Build images"
docker build -f docker/Dockerfile.api -t credit-gam-api:latest .
docker build -f docker/Dockerfile.dash -t credit-gam-dash:latest .

echo "Tag & push"
docker tag credit-gam-api:latest ${ACR_LOGIN_SERVER}/credit-gam-api:latest
docker tag credit-gam-dash:latest ${ACR_LOGIN_SERVER}/credit-gam-dash:latest

docker push ${ACR_LOGIN_SERVER}/credit-gam-api:latest
docker push ${ACR_LOGIN_SERVER}/credit-gam-dash:latest

echo "Substitute ACR in manifests"
sed "s#<ACR_LOGIN_SERVER>#${ACR_LOGIN_SERVER}#g" k8s/deployment-api.yaml > k8s/deployment-api.rendered.yaml
sed "s#<ACR_LOGIN_SERVER>#${ACR_LOGIN_SERVER}#g" k8s/deployment-dash.yaml > k8s/deployment-dash.rendered.yaml

echo "Apply manifests"
kubectl apply -f k8s/deployment-api.rendered.yaml
kubectl apply -f k8s/service-api.yaml
kubectl apply -f k8s/deployment-dash.rendered.yaml
kubectl apply -f k8s/service-dash.yaml

echo "Done."
