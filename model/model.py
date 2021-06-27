import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, Dense, Dropout, Flatten
from tensorflow.keras.activations import swish
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import Input

# Import for L2 Normalize
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


MOMENTUM = 0.999
EPSILON = 1e-3

def build_backbone(IMAGE_SIZE=[224, 224]):
    weights = "imagenet"
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=True, input_shape=(224,224, 3))
    base.summary()
    return base

def csnet_extra_model(normalizations, num_priors, num_classes=21, IMAGE_SIZE=[224, 224], convert_tfjs=False, target_transform=None):
    base = build_backbone(IMAGE_SIZE)
    # mobilenET V2
    # x2 = base.get_layer('block_6_expand_relu').output # 38x38 @ 192
    # x3 = base.get_layer('block_13_expand_relu').output # 19x19 @ 576
    # x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320

    # VGG16
    # x3 = base.get_layer('block4_conv3').output # 19x19 @ 576
    feature = base.get_layer('predictions').output # 10x10 @ 320

    x8 = Dense(128, activation='relu')(feature)
    x8 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x8)
    x8 = Dropout(0.7)(x8)

    final = Dense(4)(x8)


    return base.input, final