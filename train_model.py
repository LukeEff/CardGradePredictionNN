import copy
import csv
import os
from datetime import time

import tensorflow as tf
from tensorflow import keras
from keras.applications import resnet as resnet  # TODO add
from keras.applications import vgg16 as vgg16
from keras.applications import densenet as densenet


def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
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


def train_model(model, image, label, valid_image, valid_label, optimizer, num_epochs=25, loss='sparse_categorical_crossentropy'):
    since = time.time()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(image, label, epochs=num_epochs)
    valid_loss, valid_acc = model.evaluate(valid_image, valid_label)
    print('Validation accuracy: ', valid_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, valid_acc

def train_cnn_model(num_classes = 5, load_latest_model = False):

