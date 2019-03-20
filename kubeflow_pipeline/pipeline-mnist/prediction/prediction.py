import argparse
import json
import pickle
import pandas as pd
import tarfile
import boto3
import os

import tensorflow as tf


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess-output',
                        type=str,
                        required=True,
                        help='Preprocess output.')
    parser.add_argument('--train-output',
                        type=str,
                        required=True,
                        help='Train output.')
    parser.add_argument('--bucket-name',
                        type=str,
                        required=True,
                        help='Bucket to store confusion matrix.')
    parser.add_argument('--cm-path',
                        type=str,
                        required=True,
                        help='Path in bucket to store confusion matrix')
    return parser.parse_args()


def save_metrics(loss, acc):
    print('Saving metrics')
    metrics = {
        'metrics': [
            {
                'name': 'loss',
                'numberValue': loss,
                'format': 'RAW'
            },
            {
                'name': 'accuracy',
                'numberValue': acc,
                'format': 'PERCENTAGE'
            }
        ]
    }
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)


def save_confusion_matrix(cm_data, bucket_name, path):
    print('Saving CM')
    with open('/tmp/cm.csv', 'w') as f:
        cm_data.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)
    with tarfile.open(name='/tmp/cm.csv.tar.gz', mode='w:gz') as f:
        f.add('/tmp/cm.csv', arcname='cm.csv')

    s3_client = boto3.client(service_name='s3',
                             endpoint_url=os.environ['S3_ENDPOINT'],
                             use_ssl=False,
                             aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                             aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

    s3_client.upload_file('/tmp/cm.csv.tar.gz', bucket_name, path)

    # TODO : compute labels from cm_data['target'].distinct
    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'storage': 'minio',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': 'minio://%s/%s' % (bucket_name, path),
            'labels': list(['0', '1'])
        }]
    }

    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


def main():
    args = parse_arguments()

    preprocess_output = json.loads(args.preprocess_output)
    train_output = json.loads(args.train_output)

    print('Loading model')
    model = tf.keras.models.load_model(train_output['model'])

    print('loading prediction data')
    with open(preprocess_output['x-test'], 'rb') as f:
        x_test = pickle.load(f)

    with open(preprocess_output['y-test'], 'rb') as f:
        y_test = pickle.load(f)

    print('Compute Accuracy')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    save_metrics(test_loss, test_acc)

    print('Compute CM')
    predictions = model.predict(x_test)
    # TODO: compute CM stats (see tf.math.confusion_matrix
    data = [
        ('0', '0', 50),
        ('0', '1', 4),
        ('1', '0', 8),
        ('1', '1', 86)
    ]
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    save_confusion_matrix(df_cm, args.bucket_name, args.cm_path)


if __name__ == "__main__":
    # TODO : create bucket if not exists
    main()
