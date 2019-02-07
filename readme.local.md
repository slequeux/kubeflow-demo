```
eval $(minikube docker-env)

export KUBEFLOW_TAG=v0.4.1

curl -O https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/setup-minikube.sh
chmod +x setup-minikube.sh
```
That's all folks ! :D

http://localhost:8080/


=> Jupyter OK
- http://localhost:8080/hub

=> Image Docker simple