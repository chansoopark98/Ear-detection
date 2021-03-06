from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import os
import numpy as np


def draw_bounding(img , bboxes, img_size):
    # resizing 작업
    img_box = np.copy(img)
    for i in range(0, 54, 2):

        x1 = int(bboxes[i] * img_size[1])
        x2 = int(bboxes[i+1] * img_size[0])

        cv2.circle(img_box, (x1, x2), 5, (255, 0, 0), cv2.FILLED, cv2.LINE_4)

    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)
    cv2.imshow('test', img_box)
    cv2.waitKey()

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
        self.dataList = os.listdir(self.path_args[self.mode]+'/o_images/')

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
            load_path = self.path_args[self.mode]
            # img = image.load_img(load_path + 'o_images/' + self.mode + '_' + str(i) + '.png')
            # x = image.img_to_array(img)
            txt_path = load_path + 'o_landmarks/' + self.mode + '_' + str(i) + '.txt'
            x = cv2.imread(load_path + 'o_images/' + self.mode + '_' + str(i) + '.png')
            # image data

            height = x.shape[0]
            width = x.shape[1]

            # x1, x2, y1, y2
            with open(txt_path, 'r') as f:
                lines_list = f.readlines()
                point = []
                for index in range(3, 58):
                    tmp = lines_list[index]
                    str1, str2 = tmp.split(' ')
                    str1 = float(str1) / width
                    str2 = float(str2) / height
                    point.append(str1)
                    point.append(str2)


                point = tf.stack(point, axis=0)

                # point = tf.cast(point, dtype=tf.float32)

                draw_bounding(x, point, [height, width])

                self.y_list.append(point)

            # appending image data (preprocess)
            x = preprocess_input(x)
            x = tf.image.resize(x, [self.img_size, self.img_size])


            self.x_list.append(x)



    def get_data_len(self):
        return len(self.x_list), len(self.y_list)

    def __len__(self):
        return int(np.floor(len(self.x_list) / self.batch_size))

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
            image = self.x_list[j]
            point = self.y_list[j]

            if tf.random.uniform([]) > 0.1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 랜덤 채도
            if tf.random.uniform([]) > 0.1:
                image = tf.image.random_brightness(image, max_delta=0.15)  # 랜덤 밝기
            if tf.random.uniform([]) > 0.1:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 랜덤 대비
            if tf.random.uniform([]) > 0.1:
                image = tf.image.random_hue(image, max_delta=0.2)  # 랜덤 휴 트랜스폼
            # if tf.random.uniform([]) > 0.5: # flip
            #     image = tf.image.flip_left_right(image)
            #     point = tf.stack([1 - point[0], 1 - point[1],
            #                       1 - point[2], 1 - point[3],], axis=0)

            data.append(image)
            y_data.append(point)

        data = np.array(data)
        y_data = np.array(y_data)

        return data, y_data