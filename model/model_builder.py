import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
# from model.model import csnet_extra_model_post
from tensorflow.keras import layers

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(image_size=[224, 224]):
    normalizations = [20, 20, 20, -1, -1, -1]
    num_priors = [4, 6, 6, 6, 4, 4]
    classes = 2


    inputs, output = csnet_extra_model(normalizations, num_priors, num_classes=classes, IMAGE_SIZE=image_size,
                                       convert_tfjs=False)


    model = keras.Model(inputs, outputs=output)
    return model
