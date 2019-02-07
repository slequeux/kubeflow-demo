import shutil

import numpy as np
import tensorflow as tf

LABEL_DIMENSIONS = 10
MODEL_DIR = '/opt/model'


def cleanup_output():
    shutil.rmtree('./model', ignore_errors=True)
    shutil.rmtree(MODEL_DIR, ignore_errors=True)


def load_input():
    def preprocess_input(images, labels):
        preprocessed_images = images.astype(np.float32).reshape((len(images), 28, 28, 1))
        preprocessed_labels = tf.keras.utils.to_categorical(labels.astype(np.float32), LABEL_DIMENSIONS)
        return preprocessed_images, preprocessed_labels

    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = \
        tf.keras.datasets.mnist.load_data()
    return preprocess_input(mnist_train_images, mnist_train_labels), \
        preprocess_input(mnist_test_images, mnist_test_labels)


def input_dataset_fn(x_data, y_data):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.batch(32).repeat()
    return dataset


def build_model():
    data_input = tf.keras.layers.Input(shape=(28, 28, 1), name="inputs")
    conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(data_input)
    max_pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(max_pool1)
    max_pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    flatten = tf.keras.layers.Flatten()(max_pool2)
    softmax = tf.keras.layers.Dense(10, activation="softmax")(flatten)
    model = tf.keras.Model(inputs=data_input, outputs=softmax, name='model')

    optimizer = tf.train.AdamOptimizer()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Training mnist model using tensorflow version \'%s\'' % tf.__version__)

    cleanup_output()

    model = build_model()
    model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="./model")

    (train_images, train_labels), (test_images, test_labels) = load_input()
    model_estimator.train(input_fn=lambda: input_dataset_fn(train_images, train_labels), steps=500)

    tf.logging.info('Done training')

    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        {'inputs': tf.FixedLenFeature(shape=[28, 28, 1], dtype=np.float32)},
        default_batch_size=None
    )
    model_estimator.export_savedmodel(export_dir_base=MODEL_DIR,
                                      serving_input_receiver_fn=serving_fn,
                                      strip_default_attrs=True)

    tf.logging.info('Done exporting')
