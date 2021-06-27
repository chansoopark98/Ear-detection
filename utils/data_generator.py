from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf

import os
import pandas as pd
import random
import numpy as np

class DataGenerator(Sequence):
    def __init__(self,
                 path_args,
                 img_size,
                 batch_size: int,
                 shuffle: bool,
                 mode: str):

        self.path_args = path_args
        self.img_size = img_size
        self.mode = mode

        # train
        self.dataList = os.listdir(self.path_args[self.mode]+'/images/')

        # TODO validation and test dataset
        # -->

        self.num_classes = 2
        self.x_list = []
        self.y_list = []
        self.load_dataset()

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def load_dataset(self):
        for i, j in enumerate(self.dataList):
            # train data
            load_path = self.path_args[self.mode]
            img = image.load_img(load_path + 'images/' + self.mode + '_' + str(i) + '.png')
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = tf.image.resize(x, [self.img_size, self.img_size])

            self.x_list.append(x)

            txt_path = load_path + 'landmarks/' + self.mode + '_' + str(i) + '.txt'
            with open(txt_path, 'r') as f:
                lines_list = f.readlines()

                lines_20 = lines_list[20]
                lines_44 = lines_list[44]

                str1, str2 = lines_20.split(' ')
                str3, str4 = lines_44.split(' ')
                point = tf.stack([float(str1), float(str2), float(str3), float(str4)], axis=0)
                point = tf.cast(point, dtype=tf.float32)

                self.y_list.append(point)



    def get_data_len(self):
        return len(self.x_list), len(self.y_list)

    def __len__(self):
        return int(np.floor(len(self.x_list) / self.batch_size))

    # def __getitem__(self, index):
    #     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    #
    #     # x = [self.x_list[k] for k in indexes]
    #     # y = [self.y_list[k] for k in indexes]
    #
    #     batch_x = self.get_input(index)
    #     batch_y = self.get_target(index)
    #
    #
    #
    #     # return tuple(x), tuple(y)
    #     return ([batch_x, batch_y])


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_input(self, index):
        return self.x_list[index * self.batch_size:(index + 1) * self.batch_size]

    def get_target(self, index):
        return self.y_list[index * self.batch_size:(index + 1) * self.batch_size]

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        y_data = []
        for j in range(start, stop):
            data.append(self.x_list[j])
            y_data.append(self.y_list[j])

        # transpose list of lists
        # batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        # y_batch = [np.stack(samples, axis=0) for samples in zip(*y_data)]
        data = np.array(data)
        y_data = np.array(y_data)
        # newer version of tf/keras want batch to be in tuple rather than list
        # return tuple(batch), tuple(y_batch)
        return data, y_data