import logging

import tensorflow as tf
from tensorflow.contrib import slim

from .framework import assert_shape


LOG = logging.getLogger('main')


@slim.add_arg_scope
def gaussian_noise(inputs, scale, is_training, name=None):
    with tf.name_scope(name, 'gaussian_noise', [inputs, scale, is_training]) as scope:
        def do_add():
            noise = tf.random_normal(tf.shape(inputs))
            return inputs + noise * scale
        return tf.cond(is_training, do_add, lambda: inputs, name=scope)