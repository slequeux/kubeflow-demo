import tensorflow as tf
import pickle
import os
import json


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # METADATA
    with open('/output.txt', 'w') as f:
        f.write(json.dumps({
            'x-train': '/mnt/train/x_train.pkl',
            'y-train': '/mnt/train/y_train.pkl',
            'x-test': '/mnt/test/x_test.pkl',
            'y-test': '/mnt/test/y_test.pkl'
        }))

    # SAVE DATA

    directories = ['/mnt/train', '/mnt/test']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    with open('/mnt/train/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)

    with open('/mnt/train/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    with open('/mnt/test/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)

    with open('/mnt/test/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)


if __name__ == "__main__":
    main()
