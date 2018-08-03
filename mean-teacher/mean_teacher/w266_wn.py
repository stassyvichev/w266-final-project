import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

@add_arg_scope
def fullyConnectedLayers(h0_,hidden_dims, activation = tf.tanh, dropout_rate = 0, is_training = False, scope = None):
    with tf.variable_scope(scope, 'fullyConnectedLayers'):
        h_ = h0_
        for i, hdim in enumerate(hidden_dims):
            h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))
            if dropout_rate > 0:
                h_ = tf.layers.dropout(h_,rate=dropout_rate, training=is_training, name=("Hidden_dropout_%d"%i))
#         h_ = tf.cast(h_, dtype= tf.float32, name = ("Hidden_Layer%d"%i))
        return h_

@add_arg_scope
def dense(inputs,hdim, activation = tf.tanh, is_training = False, scope = None):
    with tf.variable_scope(scope, 'dense'):
        h_ = tf.layers.dense(inputs, hdim, activation=activation, kernel_initializer = tf.random_normal_initializer(0, 0.05), name = "hidden_1")
        return h_

@add_arg_scope
def dropoutLayer(h_, dropout_rate = 0, is_training = False, scope = None):
    with tf.variable_scope(scope, 'dropoutLayer'):
        h_ = tf.layers.dropout(h_,rate=dropout_rate, training=is_training)
        return h_
    
@add_arg_scope
def affine_layer(inputs, hidden_dim, activation_fn= None, eval_mean_ema_decay=0.999, is_training = None, scope = None, init=False):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, 'affine_layer'):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            V = tf.get_variable("V", 
                                dtype=tf.float32, 
                                shape=[inputs.get_shape()[1],hidden_dim], 
                                initializer=tf.random_normal_initializer(0, 0.05), 
                                trainable = True)
            b = tf.get_variable(name="b", 
                                dtype=tf.float32,shape=(hidden_dim), 
                                initializer=tf.zeros_initializer(), 
                                trainable = True)
            b = b.initialized_value()
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.add(tf.matmul(inputs,V_norm),b) 
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, b = [tf.get_variable(var_name) for var_name in ['V', 'b']]
            inputs = x_init = tf.add(tf.matmul(inputs,V),b)
            training_mean = tf.reduce_mean(inputs, [0])
            
            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean
            
            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs

@add_arg_scope
def affine_layer_new(inputs, hidden_dim,
                    activation_fn=None, init_scale=1., init=False,
                    eval_mean_ema_decay=0.999, is_training=None, scope=None):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, "affine_layer_new"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V',
                                [int(inputs.get_shape()[1]), hidden_dim],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            b = tf.get_variable(name="b", 
                                dtype=tf.float32,shape=(hidden_dim), 
                                initializer=tf.zeros_initializer(), 
                                trainable = True)
            
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.add(tf.matmul(inputs, V_norm),b)
            print("x_init shape:")
            print(x_init)
#             x_init = tf.add(tf.matmul(inputs,V),b) 
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
#             b = tf.get_variable('b', dtype=tf.float32,
#                                 initializer=tf.zeros_like(m_init), trainable=True)
            
            x_init = tf.reshape(
                scale_init, [1, hidden_dim]) * (x_init - tf.reshape(m_init, [1, hidden_dim]))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, g, b = [tf.get_variable(var_name) for var_name in ['V', 'g', 'b']]

            # use weight normalization (Salimans & Kingma, 2016)
            inputs = tf.matmul(inputs, V)
            training_mean = tf.reduce_mean(inputs, [0])

            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            inputs = tf.reshape(scaler, [1, hidden_dim]) * \
                inputs + tf.reshape(b, [1, hidden_dim])

            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs
        
@add_arg_scope
def fully_connected(inputs, num_outputs,
                    activation_fn=None, init_scale=1., init=False,
                    eval_mean_ema_decay=0.999, is_training=None, scope=None):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, "fully_connected"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V',
                                [int(inputs.get_shape()[1]), num_outputs],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(inputs, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros_like(m_init), trainable=True)
            x_init = tf.reshape(
                scale_init, [1, num_outputs]) * (x_init - tf.reshape(m_init, [1, num_outputs]))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, g, b = [tf.get_variable(var_name) for var_name in ['V', 'g', 'b']]

            # use weight normalization (Salimans & Kingma, 2016)
            inputs = tf.matmul(inputs, V)
            training_mean = tf.reduce_mean(inputs, [0])

            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            inputs = tf.reshape(scaler, [1, num_outputs]) * \
                inputs + tf.reshape(b, [1, num_outputs])

            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs