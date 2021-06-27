import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from metrics import CreateMetrics
from config import *
from utils.data_generator import DataGenerator
from model.model_builder import model_build

tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

config = GetConfig()

CHECKPOINT_DIR = config.save_weight
TENSORBOARD_DIR = config.tensorboard_dir

params = config.get_hyperParams()

# Create Dataset
train_generator = DataGenerator(config.get_dir_path(), params['image_size'], batch_size=params['batch_size'],
                                shuffle=True, mode='train')
valid_generator = DataGenerator(config.get_dir_path(), params['image_size'], batch_size=params['batch_size'],
                                shuffle=False, mode='test')


train_len, y_len = train_generator.get_data_len()
print("데이터 유효성 체크 : ", train_len, y_len)
valid_len, _ = valid_generator.get_data_len()
train_steps_per_epoch = train_len // params['batch_size']
valid_steps_per_epoch = valid_len // params['batch_size']

print("학습 배치 개수:", train_steps_per_epoch)
print("검증 배치 개수:", valid_steps_per_epoch)

metrics = CreateMetrics(train_generator.num_classes)


checkpoint = ModelCheckpoint(CHECKPOINT_DIR + 'ear' + '_' + 'weight_file' + '.h5',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)


polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=params['lr'],
                                                          decay_steps=params['epoch'],
                                                          end_learning_rate=params['end_lr'], power=0.5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

optimizer = tf.keras.optimizers.SGD(learning_rate=params['lr'], momentum=0.9)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

callback = [checkpoint, lr_scheduler, tensorboard]


mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

# with mirrored_strategy.scope(): # if use single gpu > with tf.device('/device:GPU:0'):
with tf.device('/device:GPU:0'):
    model = model_build(image_size=params['image_size'])

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(train_generator,
              validation_data=valid_generator,
              validation_steps=valid_steps_per_epoch,
              validation_batch_size=params['batch_size'],
              steps_per_epoch=train_steps_per_epoch,
              epochs=params['epoch'],
              callbacks=callback,
              batch_size=params['batch_size']
              )

    model.save(CHECKPOINT_DIR + 'ear' + '_' + 'model_file' + '.h5', True, True, 'h5')

