from datetime import time

import tensorflow as tf
from tensorflow import keras
from keras.applications import resnet as resnet  # TODO add
from keras.applications import vgg16 as vgg16
from keras.applications import densenet as densenet
from keras.applications import inception_v3 as inception


def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
    input_size = 0
    if model_name == 'vgg16':
        input_size = 224
        model = vgg16.VGG16(weights=None, input_size=input_size, classes=num_classes)
    elif model_name == 'densenet':
        input_size = 224
        model = densenet.DenseNet(weights=None, classes=num_classes)
    elif model_name == 'inception':
        input_size = 299
        model = inception.InceptionV3(weights=None, classes=num_classes)
    else:
        print('Invalid model name, exiting...')
        exit()

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device='cpu'):
    since = time.time()