# Prepare job

If you are using minikube, you must configure your docker CLI to use the minikube's docker API :

```bash
eval $(minikube docker-env)
```

Build the docker image :

```bash
# You currently are inside this repository
docker build -t simple-mnist:1.0.0 .
```

# Run the training

```bash
# You currently are inside this repository
kubectl create -f tfjob.yaml --namespace kubeflow
```

# Delete the training's job

```bash
# You currently are inside this repository
kubectl delete tfjob simple-mnist --namespace kubeflow
```