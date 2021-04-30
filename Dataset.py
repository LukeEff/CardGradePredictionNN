import copy
import os
import pickle
import random

import boto3 as boto3
import numpy as np
from Pillow import Image
from sklearn.model_selection import train_test_split


class PrepareDatasets:
    def __init__(self):
        self.partition_path = os.path.join('.', 'data', 'partition')
        self.make_directories()
        self.image_paths = self.get_image_paths()
        self.filter_non_baseball()
        self.filter_desired_conditions()
        self.split_train_test()

    def make_directories(self):
        if not os.path.exists(self.partition_path):
            os.mkdir(self.partition_path)

    def get_image_paths(self):
        base_path = os.path.join('.', 'data', 'imgs')  # Path to images.
        image_paths = []  # Will be an array of image paths. Example: image_paths[0] = ./data/imgs/bar.jpg

        for dir_name in os.listdir(base_path):
            dir_path = os.path.join(base_path, dir_name)
            for img_filename in os.listdir(dir_path):
                image_path = os.path.join(dir_name, img_filename)
                image_paths.append(image_path)
        random.shuffle(image_paths)
        return image_paths

    def filter_non_baseball(self):
        baseball_sport = '185223'  # Not sure why this is here. Perhaps an identifier related to search name...
        self.image_paths = [path for path in self.image_paths if baseball_sport in path]

    """
    Exclude good and fair cards to create a 9, 7, 5, 3, 1 scale.
    """

    def filter_desired_conditions(self):
        self.image_paths = [path for path in self.image_paths if "GOOD" not in path and "FAIR" not in path]

    """
    Only retain images that can be loaded successfully as a (255, 255, 3) image.
    """

    def remove_bad_imagaes(self):
        old_image_paths = copy.copy(self.image_paths)
        image_paths = []
        for path in old_image_paths:
            try:
                full_path = os.path.join("..", "data", "imgs", path)
                img = Image.open(full_path)
                img = np.array(img)
                assert img.shape[2] == 3
                image_paths.append(path)

            except Exception as e:
                print("Skip bad image {}".format(path))
                print(e)

    def split_train_test(self, test_size=0.1):
        train_paths, test_paths = train_test_split(self.image_paths, test_size=test_size)
        partition = {
            'train': train_paths,
            'test': test_paths
        }
        self.pickle_partition(partition)

    """
    Pickle partition object
    """

    def pickle_partition(self, partition):
        pickle_partition_path = os.path.join(self.partition_path, "partition.p")
        with open(pickle_partition_path, 'wb') as obj:
            pickle.dump(partition, obj)
        print("wrote {}".format(pickle_partition_path))

    """
    Create labels dictionary for our dataset for the training of the model
    """

    def create_labels_dictionary(self):
        labels = ['MINT', 'NM', 'EX', 'VG', 'POOR']
        label_to_index = {label: i for i, label in enumerate(labels)}
        filename_to_label = {}
        for path in self.image_paths:
            label = path.split("_")[-1].lstrip("condition").rstrip(".jpg")
            int_label = label_to_index[label]
            filename_to_label[path] = int_label

            pickle_partition_path = os.path.join(self.partition_path, "labels.p")
            with open(pickle_partition_path, "wb") as obj:
                pickle.dump(filename_to_label, obj)
            print("wrote {}".format(pickle_partition_path))


"""
Iterable Dataset class allowing streaming large datasets rather than loading into RAM.
"""


class Dataset:
    def __init__(self, list_IDs, labels, apply_rotations=False):
        """
        parameters
        ----------
        list_IDs (str []) file paths.
        labels (str []) target labels.
        apply_rotations (bool) : if True then randomly rotate images.
            NOTE! This should be FALSE for testing data.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.apply_rotations = apply_rotations
        self.base_path = os.path.join('.', 'data', 'ml_images')
        self.s3_resource = boto3.resource('s3')

    def make_directories(self):
        conditions = ["MINT", "NM", "EX", "VG", "FAIR", "GOOD", "POOR"]
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
            for condition in conditions:
                path = os.path.join(self.base_path, condition)
                os.mkdir(path)

    def __len__(self):
        return len(self.list_IDs)

    """
    Download file name from Amazonrew s3 as ./data/ml_images/{REMOTE FILENAME}
    Download image from s3 as img.jpg
    """

    def download_from_s3(self, remote_filename):
        local_filename = os.path.join('.', 'data', 'ml_images', remote_filename)
        print("download {}".format(remote_filename))

        resp = self.s3_resource.Object(
            'autonize',
            remote_filename
        ).download_file(local_filename)
        return None

    def load_and_preprocess_image(self, image_path, apply_rotations=False):
        image = Image.open(image_path)
        image = image.resize((255, 255), Image.ANTIALIAS)

        if apply_rotations:
            image = image  # TODO implement image rotation method

        x = np.array(image)
        x = self.normalize(x)
        y = 0
        return x, y

    def normalize(self, x):
        return x / 255

    def __getitem__(self, index):
        try:
            remote_path = self.list_IDs[index]
            y = self.labels[remote_path]
            local_path = os.path.join('.', 'data', 'ml_images', remote_path)
            if not os.path.exists(local_path):
                self.download_from_s3(remote_path)
        except Exception as e:
            print(e)
            print('Exception loading data... using random image, label instead.')
            x = np.random.random((255, 255, 3))
            y = 0

        return x, y