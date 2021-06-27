from utils.priors import *
import os
import time

INPUT_SIZE = [224, 224]
iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

hyper_params = {
                "epoch": 1,
                "lr": 0.0001,
                "batch_size": 1
                }

def set_priorBox():
    return [
        Spec(38, 8, BoxSizes(9, 12), [2]),
        Spec(19, 16, BoxSizes(23, 33), [2, 3]),
        Spec(10, 32, BoxSizes(54, 108), [2, 3]),
        Spec(5, 64, BoxSizes(113, 134), [2, 3]),
        Spec(3, 100, BoxSizes(182, 226), [2]),
        Spec(1, 300, BoxSizes(264, 315), [2])
    ]

class GetConfig:
    def __init__(self,
                 data_dir='datasets/',
                 result_dir='checkpoints/'):

        # set dir
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.time = self.get_current_time()
        self.train_dir = os.path.join(data_dir + 'train/')
        self.valid_dir = os.path.join(data_dir + 'test/')
        self.test_dir = os.path.join(data_dir + 'test/')
        self.result_dir = result_dir + self.time
        self.tensorboard_dir = result_dir + self.time + '/tensorboard'

        # save file dir
        self.save_weight = os.path.join(self.result_dir + '/save_weights/')
        self.save_backup = os.path.join(self.result_dir + '/back_weights/')
        self.test_result = os.path.join(self.result_dir + '/test/')

        # create dir
        self.create_directory()

        # train params
        self.args = hyper_params



    def create_directory(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.result_dir + self.time, exist_ok=True)

        os.makedirs(self.save_weight, exist_ok=True)
        os.makedirs(self.save_backup, exist_ok=True)
        os.makedirs(self.test_result, exist_ok=True)

    def get_current_time(self):
        return str(time.strftime('%m%d', time.localtime(time.time())))

    def get_dir_path(self):
        return {
            "train": self.train_dir,
            "validation": self.valid_dir,
            "test": self.test_dir,
            "result": self.result_dir,
            "tensorboard": self.tensorboard_dir
        }

    def get_hyperParams(self):
        return self.args


