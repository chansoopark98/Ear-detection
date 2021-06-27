from tensorflow.keras.utils import Sequence
import os
import pandas as pd
import random
import numpy as np

class DataGenerator(Sequence):
    def __init__(self,
                 path_args,
                 batch_size: int,
                 shuffle: bool,
                 mode: str):

        self.path_args = path_args
        self.mode = mode

        # train
        self.dataList = os.listdir(self.path_args[self.mode])

        # TODO validation and test dataset
        # -->

        self.timeSteps = 100
        self.num_classes = 5
        self.x_list = []
        self.y_list = []
        self.load_dataset()

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def load_dataset(self):
        for i, j in enumerate(self.dataList):

            data = pd.read_csv(self.path_args[self.mode] + j).to_numpy()
            # data.shape = (549, 2)
            t, f = data.shape  # t: 549, f: 2

            start = random.randint(0, t - self.timeSteps - 1)
            input_data = np.reshape(data[start: start + self.timeSteps, 0], (1, self.timeSteps, 1))
            targets = np.reshape(data[start: start + self.timeSteps, 1], (-1)).astype(np.int)
            result_data = np.eye(self.num_classes)[targets]
            result_data = np.reshape(result_data, [1, self.timeSteps, self.num_classes])

            # self.data_list.append([input_data, result_data.astype(np.float)])
            self.x_list.append(input_data)
            self.y_list.append(result_data.astype(np.float))

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
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        y_batch = [np.stack(samples, axis=0) for samples in zip(*y_data)]

        # newer version of tf/keras want batch to be in tuple rather than list
        return tuple(batch), tuple(y_batch)