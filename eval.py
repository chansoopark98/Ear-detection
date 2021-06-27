import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config import *
from utils.data_generator import DataGenerator
from model.model_builder import model_build
import tqdm

tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

config = GetConfig()

CHECKPOINT_DIR = config.save_weight
TENSORBOARD_DIR = config.tensorboard_dir
WEIGHT_PATH = None

params = config.get_hyperParams()


test_generator = DataGenerator(config.get_dir_path(), params['image_size'], batch_size=params['batch_size'],
                                shuffle=False, mode='test')
test_len, _ = test_generator.get_data_len()
test_steps = test_len // params['batch_size']


model = model_build(image_size=params['image_size'])
model.load_weights(WEIGHT_PATH)

# TODO Evaluation 작업 해야 함
with tf.device('/device:GPU:0'):
    print("Evaluating..")
    for x, y in tqdm(test_generator, total=test_steps):
        pred = model.predict_on_batch(x)
        print(pred)





