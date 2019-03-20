FROM tensorflow/tensorflow:1.12.0

RUN mkdir /opt/tfjob

ADD mnist.py /opt/tfjob
ADD patch/parameter_server_strategy.patch.min.py /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/distribute/python/parameter_server_strategy.py

WORKDIR /opt/tfjob
ENTRYPOINT /usr/bin/python /opt/tfjob/mnist.py
