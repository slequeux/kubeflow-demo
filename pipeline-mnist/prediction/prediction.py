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
    parser.add_argument('--train-output',
                        type=str,
                        required=True,
                        help='Train output.')

    return parser.parse_args()


def main():
    args = parse_arguments()

    preprocess_output = json.loads(args.preprocess_output)
    train_output = json.loads(args.train_output)

    model = tf.keras.models.load_model(train_output['model'])

    with open(preprocess_output['x-test'], 'rb') as f:
        x_test = pickle.load(f)

    with open(preprocess_output['y-test'], 'rb') as f:
        y_test = pickle.load(f)

    results = model.evaluate(x_test, y_test)

    print(results)

    # TODO: Save output

    with open('/output.txt', 'w') as f:
        f.write(json.dumps({

        }))


if __name__ == "__main__":
    main()
