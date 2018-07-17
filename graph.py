import tensorflow as tf

def affineLayer(hidden_dims, x):
    W = tf.get_variable("W", shape=[x.get_shape().as_list()[1], hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name="b", shape=(hidden_dim), initializer=tf.zeros_initializer())
    return tf.add(tf.matmul(x,W),b)

def newFunc():
    pass 
