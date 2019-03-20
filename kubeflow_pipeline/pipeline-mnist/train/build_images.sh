#!/usr/bin/env bash

eval $(minikube docker-env)
docker build -t romibuzi/kubeflow-mnist:train-seventh .