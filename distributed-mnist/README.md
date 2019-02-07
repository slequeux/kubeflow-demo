# Prepare job

If you are using minikube, you must configure your docker CLI to use the minikube's docker API :

```bash
eval $(minikube docker-env)
```

Build the docker image :

```bash
# You currently are inside this repository
docker build -t distributed-mnist:1.0.0 .
```

# Run the training

```bash
# You currently are inside this repository
kubectl create -f tfjob.yaml --namespace kubeflow
```

# Delete the training's job

```bash
# You currently are inside this repository
kubectl delete tfjob distributed-mnist --namespace kubeflow
```

# Known issues

Currently, when running this code inside my environment, I got the following error :
```python
Traceback (most recent call last):
  File "/opt/tfjob/mnist.py", line 53, in <module>
    model_estimator.train(input_fn=lambda: input_dataset_fn(train_images, train_labels), steps=2000)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 354, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 1205, in _train_model
    return self._train_model_distributed(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 1321, in _train_model_distributed
    destinations='/device:CPU:0'))[0]
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/training/distribute.py", line 751, in reduce
    return self._reduce(aggregation, value, destinations)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/distribute/python/parameter_server_strategy.py", line 305, in _reduce
    self._verify_destinations_not_different_worker(destinations)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/distribute/python/parameter_server_strategy.py", line 302, in _verify_destinations_not_different_worker
    (d, self._worker_device))
AttributeError: 'ParameterServerStrategy' object has no attribute '_worker_device'
```

Patch to `parameter_server_strategy.py` applied to avoid this.