from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.applications import mobilenet_v2, ResNet50V2

MOMENTUM = 0.999
EPSILON = 1e-3

def build_backbone(IMAGE_SIZE=224):
    # weights = "imagenet"
    weights = None
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=True, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # base = ResNet50V2(include_top=True, weights=weights, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.summary()
    return base

def csnet_extra_model(IMAGE_SIZE=224):
    base = build_backbone(IMAGE_SIZE)

    feature = base.get_layer('predictions').output

    x8 = Dense(128, activation='relu')(feature)
    x8 = Dropout(0.7)(x8)
    final = Dense(4)(x8)

    return base.input, final