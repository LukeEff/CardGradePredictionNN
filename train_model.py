import copy
import csv
import os
import pickle
import time

import tensorflow as tf
from tensorflow import keras
from keras.applications import resnet as resnet  # TODO add
from keras.applications import vgg16 as vgg16
from keras.applications import densenet as densenet

import dataset as ds


def init_model(model_name, num_classes, use_pretrained=True):
    input_size = 0
    if model_name == 'vgg16':
        input_size = 224
        model = vgg16.VGG16(weights=None, input_size=input_size, classes=num_classes)
    elif model_name == 'densenet':
        input_size = 224
        model = densenet.DenseNet(weights=None, classes=num_classes)
    else:
        print('Invalid model name, exiting...')
        exit()
    return model, input_size


def train_model(model,
                train_dataset,
                valid_dataset,
                optimizer='adam',
                num_epochs=25,
                loss='sparse_categorical_crossentropy'):
    since = time.time()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_dataset, validation_data=valid_dataset, epochs=num_epochs, validation_split=0.2)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, model.history

def train_cnn_model(num_classes = 5, load_latest_model = False):

    if load_latest_model:
        print('Load pickled latest_model.p')
        model_path = os.path.join('.', 'models', 'latest_model.p')
        model = pickle.load(open(model_path, 'rb'))
    else:
        model, _ = init_model(
            model_name='vgg16',
            num_classes=num_classes,
            use_pretrained=True
        )
    train_dataset = ds.init_dataset('train')
    validation_dataset = ds.init_dataset('train')
    model, history = train_model(model=model, train_dataset=train_dataset, valid_dataset=validation_dataset)

    model.save(os.path.join(".", "models", "model_in_training"))
    # Pickle best performing model.
    open_file_path = os.path.join(".", "models", "best_model.p")
    with open(open_file_path, "wb") as open_file:
        pickle.dump(model, open_file)
    print("wrote {}".format(open_file_path))

    # Pickle history of best performing model.
    open_file_path = os.path.join(".", "models", "history.p")
    with open(open_file_path, "wb") as open_file:
        pickle.dump(history, open_file)
    print("wrote {}".format(open_file_path))

