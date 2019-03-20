#!/usr/bin/env bash

docker build -t romibuzi/kubeflow-mnist:train-seventh .
docker push romibuzi/kubeflow-mnist:train-seventh
