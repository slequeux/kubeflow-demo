import argparse
import json
import os
import pickle
from datetime import datetime

import tensorflow as tf


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess-output',
                        type=str,
                        required=True,
                        help='Preprocess output.')
    parser.add_argument('--epoch',
                        type=int,
                        required=False,
                        default=5,
                        help='Number of epoch training')
    parser.add_argument('--hidden-layer-size',
                        type=int,
                        required=False,
                        default=512,
                        help='Number of neurons in hidden layer')
    parser.add_argument('--dropout',
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout')

    return parser.parse_args()


def main():
    args = parse_arguments()
    preprocess_output = json.loads(args.preprocess_output)

    with open(preprocess_output['x-train'], 'rb') as f:
        x_train = pickle.load(f)

    with open(preprocess_output['y-train'], 'rb') as f:
        y_train = pickle.load(f)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu),
        tf.keras.layers.Dropout(args.dropout),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=args.epoch)

    model_serving_dir = "/mnt/mnist_model"
    now = datetime.now().strftime("%Y%m%d%H%I%S")
    if not os.path.exists(model_serving_dir):
        os.makedirs(model_serving_dir)

    model.save('/mnt/model.h5')

    # Save  model for tf-serving
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            os.path.join(model_serving_dir, now),
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})

    with open('/output.txt', 'w') as f:
        f.write(json.dumps({
            'keras_model': '/mnt/model.h5',
            'serving_model_dir': model_serving_dir
        }))


if __name__ == "__main__":
    main()
