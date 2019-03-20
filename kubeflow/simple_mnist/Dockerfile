FROM tensorflow/tensorflow:1.12.0

RUN mkdir /opt/tfjob

ADD mnist.py /opt/tfjob
WORKDIR /opt/tfjob
ENTRYPOINT /usr/bin/python /opt/tfjob/mnist.py
