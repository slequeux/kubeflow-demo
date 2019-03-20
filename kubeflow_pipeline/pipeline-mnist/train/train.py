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
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),  # TODO : Parametrize dropout
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1)

    model.save('/mnt/model.h5')

    with open('/output.txt', 'w') as f:
        f.write(json.dumps({
            'model': '/mnt/model.h5',
        }))


if __name__ == "__main__":
    main()
