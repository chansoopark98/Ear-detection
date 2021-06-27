from tensorflow import keras
from model.model import csnet_extra_model

def model_build(image_size=224):
    inputs, output = csnet_extra_model(IMAGE_SIZE=image_size)
    model = keras.Model(inputs, outputs=output)
    return model
