#!/usr/bin/env bash

eval $(minikube docker-env)
docker build -t romibuzi/kubeflow-mnist:deploy-0.0.1 .
