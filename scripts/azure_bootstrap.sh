#!/usr/bin/env bash
set -euo pipefail

: "${AZ_SUBSCRIPTION:?}"
: "${AZ_REGION:?}"
: "${AZ_RESOURCE_GROUP:?}"
: "${ACR_NAME:?}"
: "${AKS_NAME:?}"
: "${EH_NAMESPACE:?}"
: "${EH_NAME:?}"

az account set --subscription "$AZ_SUBSCRIPTION"

echo ">> Resource Group"
az group create -n "$AZ_RESOURCE_GROUP" -l "$AZ_REGION"

echo ">> Azure Container Registry"
az acr create -n "$ACR_NAME" -g "$AZ_RESOURCE_GROUP" --sku Basic
ACR_LOGIN_SERVER=$(az acr show -n "$ACR_NAME" -g "$AZ_RESOURCE_GROUP" --query loginServer -o tsv)
echo "ACR: $ACR_LOGIN_SERVER"

echo ">> AKS Cluster (attach ACR)"
az aks create -n "$AKS_NAME" -g "$AZ_RESOURCE_GROUP" --node-count 2 --generate-ssh-keys --attach-acr "$ACR_NAME"
az aks get-credentials -n "$AKS_NAME" -g "$AZ_RESOURCE_GROUP"

echo ">> Event Hubs (Kafka-compatible)"
az eventhubs namespace create -n "$EH_NAMESPACE" -g "$AZ_RESOURCE_GROUP" -l "$AZ_REGION" --enable-kafka true
az eventhubs eventhub create -n "$EH_NAME" --namespace-name "$EH_NAMESPACE" -g "$AZ_RESOURCE_GROUP" --partition-count 2 --message-retention 1
echo "Event Hubs Kafka bootstrap: $EH_NAMESPACE.servicebus.windows.net:9093"

echo ">> Done. Export ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER"
