import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config import *
from model.model_builder import model_build
from tqdm import tqdm
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
import numpy as np

BATCH_SIZE = 1
INPUT_DIR = './inputs/'
OUTPUT_DIR = './outputs'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

config = GetConfig()

CHECKPOINT_DIR = config.save_weight
TENSORBOARD_DIR = config.tensorboard_dir
WEIGHT_PATH = './checkpoints/0628/save_weights/ear_weight_file.h5'

params = config.get_hyperParams()

def prepare_for_prediction(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img, [params['image_size'], params['image_size']])
    img = preprocess_input(img)
    return img

def decode_img(img, image_size):
    # 텐서 변환
    img = tf.image.decode_png(img, channels=3)
    # 이미지 리사이징
    return tf.image.resize(img, image_size)

def draw_bounding(img , bboxes, img_size):
    img_box = np.copy(img)
    for i in range(0, 54, 2):

        x1 = int(bboxes[i] * img_size[1])
        x2 = int(bboxes[i+1] * img_size[0])

        cv2.circle(img_box, (x1, x2), 5, (255, 0, 0), cv2.FILLED, cv2.LINE_4)

    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)




model = model_build(image_size=params['image_size'])
model.load_weights(WEIGHT_PATH)
test_steps = 105 // BATCH_SIZE + 1
filenames = os.listdir(INPUT_DIR)
filenames.sort()
dataset = tf.data.Dataset.list_files(INPUT_DIR + '*', shuffle=False)
dataset = dataset.map(prepare_for_prediction)
dataset = dataset.batch(BATCH_SIZE)
x, y = 0, BATCH_SIZE
# TODO Evaluation 작업 해야 함
with tf.device('/device:GPU:0'):
    print("Evaluating..")
    for batch in tqdm(dataset, total=test_steps):

        pred = model.predict_on_batch(batch)

        for i, path in enumerate(filenames[x:y]):

            im = cv2.imread(INPUT_DIR + '/' + path)
            # [x1, x2, y1, y2]
            draw_bounding(im, pred[i], img_size=im.shape[:2])
            fn = OUTPUT_DIR + '/' + path + '.jpg'
            cv2.imwrite(fn, im)

        x = y
        y += BATCH_SIZE

