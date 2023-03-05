import numpy as np
import os
import pandas as pd
import random
from skimage.io import imread
import tensorflow as tf


class BaseLoader(object):

    def __init__(self,
                 train_data_dir,
                 test_data_dir=None,
                 train_percent=0.8,
                 train_idx=None,
                 test_idx=None,
                 transfer_model=None,
                 max_n=None,
                 batch_size=32,
                 shuffle=True,
                 one_hot=True,
                 random_seed=42):
        super(BaseLoader, self).__init__()

        tf.random.set_seed(random_seed)

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_percent = train_percent
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.transfer_model = transfer_model
        self.transfer_model.trainable = False
        self.max_n = max_n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.one_hot = one_hot

        if self.test_data_dir is None:
            self.data_dir = self.train_data_dir
        else:
            self.data_dir = None
        
        self.raw_data = None
        self.train = None
        self.test = None
        self.num_classes = None

        self.keras_data = {
            'cifar10': tf.keras.datasets.cifar10.load_data
        }

    def load(self):
        if self.test_data_dir is None:
            self._load_without_separate_test_data()
        elif os.path.isdir(self.test_data_dir):
            self._load_with_separate_test_data()

    def _load_without_separate_test_data(self):
        if self.data_dir in self.keras_data.keys():
            train, test = self.load_data_from_keras()
        elif os.path.isdir(self.data_dir):
            train, test = self.load_data_from_files()
        else:
            raise TypeError

        if self.shuffle:
            train = train.shuffle(int(1e5))
            test = test.shuffle(int(1e5))

        train = train.batch(self.batch_size).repeat(1)
        test = test.batch(self.batch_size).repeat(1)

        self.train = train
        self.test = test

    def _load_with_separate_test_data(self):
        if os.path.isdir(self.train_data_dir) and os.path.isdir(self.test_data_dir):
            train = self.load_data_from_files(has_separate_test_data=True, mode="train")
            test = self.load_data_from_files(has_separate_test_data=True, mode="test")
            if self.shuffle:
                train = train.shuffle(int(1e5))
                test = test.shuffle(int(1e5))

            train = train.batch(self.batch_size).repeat(1)
            test = test.batch(self.batch_size).repeat(1)

            self.train = train
            self.test = test

    def load_data_from_files(self, mode="train", has_separate_test_data=False):
        """Load JPG files from data directory.

        Expects subdirectories to be class labels. For example:

        - data_dir
            - CLASS A
                - <>.jpg
                - <>.jpg
            - CLASS B
                - <>.jpg
                - <>.jpg
        """
        reformat = lambda x: tf.keras.applications.resnet_v2.preprocess_input(x)
        reshape = lambda x: np.array([_x for _x in x])

        if mode == "train":
            self.data_dir = self.train_data_dir
        elif mode == "test":
            self.data_dir = self.test_data_dir

        classes = os.listdir(self.data_dir)
        self.num_classes = len(classes)

        data = []
        for cls in classes:
            if cls is not "bottlenecks":
                fnames = [os.path.join(self.data_dir, cls, f) for f in os.listdir(os.path.join(self.data_dir, cls))
                          if f.endswith(".jpg")]
                if self.max_n is not None:
                    random.shuffle(fnames)
                    fnames = fnames[:self.max_n]

                for fname in fnames:
                    _img = reformat(imread(fname))
                    _img = tf.image.resize(_img, [224, 224])
                    data.append({"class": cls, "img": _img, "fname": fname})

        data = pd.DataFrame(data)
        self.raw_data = data

        if not has_separate_test_data:
            cardinality = len(data["class"])
            if self.train_idx is not None and self.test_idx is not None:
                train_idx = self.train_idx
                test_idx = self.test_idx
            else:
                train_idx = np.random.choice(list(range(cardinality)), int(self.train_percent * cardinality))
                test_idx = [i for i in range(cardinality) if i not in train_idx]

            X_train = data["img"].values[train_idx]
            y_train = pd.get_dummies(data["class"].values[train_idx]).values

            X_test = data["img"].values[test_idx]
            y_test = pd.get_dummies(data["class"].values[test_idx]).values

            if self.transfer_model is not None:
                X_train = self.transfer_model(reshape(X_train))
                X_train = np.array([x.numpy()[0, 0, :].reshape(1,-1) for x in X_train])

                X_test = self.transfer_model(reshape(X_test))
                X_test = np.array([x.numpy()[0, 0, :].reshape(1,-1) for x in X_test])

            else:
                X_train = reshape(X_train)
                X_test = reshape(X_test)

            train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

            return train, test

        elif has_separate_test_data:
            X = data["img"].values
            if self.one_hot:
                y = pd.get_dummies(data["class"]).values.astype(np.float32)
            elif not self.one_hot:
                y, _ = pd.factorize(data["class"])

            if self.transfer_model is not None:
                X = self.transfer_model(reshape(X))
                X = np.array([x.numpy()[0, 0, :] for x in X])

            else:
                X = reshape(X)

            ds = tf.data.Dataset.from_tensor_slices((X, y))
            return ds

    def load_data_from_keras(self):
        """Load data from keras datasets"""
        _train, _test = self.keras_data[self.data_dir]()

        reformat = lambda x: tf.keras.applications.resnet.preprocess_input(x)
        one_hot = lambda x: pd.get_dummies(x[1].reshape(-1)).values

        if self.transfer_model is not None:
            train = (tf.squeeze(self.transfer_model(reformat(_train[0]))), one_hot(_train))
            test = (tf.squeeze(self.transfer_model(reformat(_test[0]))), one_hot(_test))
        else:
            train = _train
            test = _test

        return tf.data.Dataset.from_tensor_slices(train), tf.data.Dataset.from_tensor_slices(test)

class ACPMRILite(BaseLoader):

    def __init__(self):
        super(ACPMRILite, self).__init__(
            train_data_dir="../../data/MRI224_aug1000",
            test_data_dir="../../data/MRI_VAL",
            train_percent=0.8,
            transfer_model=tf.keras.applications.ResNet152V2(
                input_shape=(224,224,3), include_top=False, weights='imagenet'),
            max_n=200,
            batch_size=8,
            shuffle=True,
            random_seed=42)


class ACPCTLite(BaseLoader):

    def __init__(self):
        super(ACPCTLite, self).__init__(
            train_data_dir="../../data/CT224_aug100",
            test_data_dir="../../data/CT_VAL",
            train_percent=0.8,
            transfer_model=tf.keras.applications.ResNet152V2(
                input_shape=(224,224,3), include_top=False, weights='imagenet'),
            max_n=200,
            batch_size=8,
            shuffle=True,
            random_seed=42)