#!/usr/bin/env bash

eval $(minikube docker-env)
docker build -t romibuzi/kubeflow-mnist:preprocessing-0.0.1 .
