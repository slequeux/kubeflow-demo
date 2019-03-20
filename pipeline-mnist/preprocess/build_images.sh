#!/usr/bin/env bash

docker build -t romibuzi/kubeflow-mnist:preprocessing-fifth .
docker push romibuzi/kubeflow-mnist:preprocessing-fifth
