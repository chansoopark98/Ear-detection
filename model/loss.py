import tensorflow as tf

def total_loss(y_true, y_pred):
    return tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.AUTO)(y_true,y_pred)


def smooth_l1(labels, scores, sigma=1.0):
    diff = scores - labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1 / (sigma ** 2)), 0.5 * (sigma * diff) ** 2, abs_diff - 1 / (2 * sigma ** 2))