#!/usr/bin/env bash

docker build -t romibuzi/kubeflow-mnist:prediction-third .
docker push romibuzi/kubeflow-mnist:prediction-third
