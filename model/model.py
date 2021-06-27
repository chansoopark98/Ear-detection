from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import Input

MOMENTUM = 0.999
EPSILON = 1e-3

def build_backbone(IMAGE_SIZE=224):
    weights = "imagenet"
    # weights = None
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=True, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # base = ResNet50V2(include_top=True, weights=weights, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.summary()
    return base

def csnet_extra_model(IMAGE_SIZE=224):
    # base = build_backbone(IMAGE_SIZE)
    #
    # feature = base.get_layer('predictions').output
    #
    # x8 = Dense(128, activation='relu')(feature)
    # x8 = Dropout(0.2)(x8)
    # final = Dense(4)(x8)
    base_channel = 8
    input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    conv0 = Conv2D(base_channel, (3, 3), kernel_initializer='random_uniform', activation='relu')(input)

    # conv 여러개 넣을 경우 속성에 padding = "same" 추가
    conv1 = Conv2D(base_channel * 2, (3, 3), activation='relu')(conv0)
    mxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(base_channel * 4, (3, 3), activation='relu')(mxpool1)
    mxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(base_channel * 8, (3, 3), activation='relu')(mxpool2)
    norm1 = BatchNormalization()(conv3)
    mxpool3 = MaxPooling2D(pool_size=(2, 2))(norm1)
    drop1 = Dropout(0.3)(mxpool3)

    conv4 = Conv2D(base_channel * 16, (5, 5), activation='relu')(drop1)
    norm2 = BatchNormalization()(conv4)
    mxpool4 = MaxPooling2D(pool_size=(2, 2))(norm2)

    conv5 = Conv2D(base_channel * 32, (5, 5), activation='relu')(mxpool4)
    norm3 = BatchNormalization()(conv5)
    mxpool5 = MaxPooling2D(pool_size=(2, 2))(norm3)
    drop2 = Dropout(0.5)(mxpool5)

    flat = Flatten()(drop2)

    dense1 = Dense(base_channel * 64, activation='relu')(flat)
    norm4 = BatchNormalization()(dense1)
    drop3 = Dropout(0.7)(norm4)

    dense2 = Dense(4, activation='sigmoid')(drop3)

    return input, dense2