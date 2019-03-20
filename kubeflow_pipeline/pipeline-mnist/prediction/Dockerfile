FROM tensorflow/tensorflow:1.12.0

ADD prediction.py /prediction.py

RUN pip install pandas
RUN pip install boto3

ENTRYPOINT ["python", "/prediction.py"]