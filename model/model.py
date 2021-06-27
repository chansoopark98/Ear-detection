from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.applications import mobilenet_v2

MOMENTUM = 0.999
EPSILON = 1e-3

def build_backbone(IMAGE_SIZE=224):
    weights = "imagenet"
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=True, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    return base

def csnet_extra_model(IMAGE_SIZE=224):
    base = build_backbone(IMAGE_SIZE)

    feature = base.get_layer('predictions').output

    x8 = Dense(128, activation='relu')(feature)
    x8 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x8)
    x8 = Dropout(0.7)(x8)

    final = Dense(4)(x8)

    return base.input, final