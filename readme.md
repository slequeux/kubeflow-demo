# Kubeflow

## Install Kubernetes cluster

```bash
minikube start
# 2 CPU
# 8G RAM
# 50G disk

eval $(minikube docker-env)
```

## Install Kubeflow

To download and install
```
export KUBEFLOW_TAG=v0.4.1

curl -O https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/setup-minikube.sh
chmod +x setup-minikube.sh
```

To use already downloaded script
```bash
cd setup
./setup-minikube.sh

kubectl -n kubeflow get all
```

Kubeflow available at http://localhost:8080

## Storage

Create storage for models
```bash
cd storage
kubectl create -f model_storage.yaml

kubectl -n kubeflow get pv
kubectl -n kubeflow get pvc
``` 

## Explore Jupyter

http://localhost:8080/hub

## Run simple tensorflow script

See [simple mnist](./simple_mnist/README.md)

## Run distributed tensorflow script

See [distributed mnist](./distributed-mnist/README.md)

## Deploy model

```bash
APP_NAME=???

ks init $APP_NAME
cd $APP_NAME

ks env set default --namespace kubeflow
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/master/kubeflow
ks pkg install kubeflow/tf-serving

ks generate io.ksonnet.pkg.tf-serving my-serving-comp
ks param set my-serving-comp deployHttpProxy True
ks param set my-serving-comp modelStorageType nfs
ks param set my-serving-comp modelPath /mnt
ks param set my-serving-comp nfsPVC model-pv-claim
ks param set my-serving-comp version 1549557994 --as-string
ks param list

ks apply default -c my-serving-comp
```