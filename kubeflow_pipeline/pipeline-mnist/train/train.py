import argparse
import json
import pickle

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

    model.save('/mnt/model.h5')

    with open('/output.txt', 'w') as f:
        f.write(json.dumps({
            'model': '/mnt/model.h5',
        }))


if __name__ == "__main__":
    main()
