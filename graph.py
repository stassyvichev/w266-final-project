import tensorflow as tf

def fullyConnectedLayers(h0_,hidden_dims, activation = tf.tanh, dropout_rate = 0, is_training = False):
    h_ = h0_
    for i, hdim in enumerate(hidden_dims):
        h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))
        if dropout_rate > 0:
            h_ = tf.layers.dropout(h_,rate=dropout_rate, training=is_training)
    return h_

def makeLogits(h_, num_classes):
    with tf.variable_scope("Logits"):
#         h_ = tf.cast(h_, dtype= tf.float32)
#         print(h_)
        W_out_ = tf.get_variable("W_out", shape = [h_.get_shape().as_list()[1], num_classes], initializer=tf.random_normal_initializer(dtype= tf.float32))
        b_out_ = tf.get_variable("b_out", shape = [num_classes], initializer = tf.zeros_initializer())
#         print(b_out_)
        logits_ = tf.matmul(h_,W_out_) + b_out_
        return logits_

def makeLoss(logits_, labels_):
    with tf.name_scope("Softmax"):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=labels_))

def softmaxOutputLayer(h_, labels_, num_classes):
    logits_ = makeLogits(h_, num_classes)
    
    if labels_ is None:
        return None, logits_
    
    loss_ = makeLoss(logits_, labels_)
    return loss_, logits_

def classifierModelFn(features, labels, mode, params):
    # Seed the RNG for repeatability
    tf.set_random_seed(params.get('rseed', 42))
#     print(labels)
    # Check if this graph is going to be used for training.
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope("Encoder"):
        xs_ = features["x"]
        # we already have our encodings, so the first layer is our input
        # Build a stack of fully-connected layers
        h_ = fullyConnectedLayers(xs_, params['hidden_dims'], activation=tf.tanh,
                           dropout_rate=params['dropout_rate'], is_training=is_training)
        h_ = tf.cast(h_, dtype= tf.float32)
    # Construct softmax layer and loss functions
    with tf.variable_scope("Output_Layer"):
        ce_loss_, logits_ = softmaxOutputLayer(h_, labels, params['num_classes'])
    
    # Some code for handling prediction
    with tf.name_scope("Prediction"):
        pred_proba_ = tf.nn.softmax(logits_, name="pred_proba")
        pred_max_ = tf.argmax(logits_, 1, name="pred_max")
        predictions_dict = {"proba": pred_proba_, "max": pred_max_}

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If predict mode, don't bother computing loss.
        return tf.estimator.EstimatorSpec(mode=mode,  predictions=predictions_dict)
    
    # L2 regularization (weight decay) on parameters, from all layers
    with tf.variable_scope("Regularization"):
        l2_penalty_ = tf.nn.l2_loss(xs_)  # l2 loss on embeddings
        for var_ in tf.trainable_variables():
            if "Embedding_Layer" in var_.name:
                continue
            l2_penalty_ += tf.nn.l2_loss(var_)
        l2_penalty_ *= params['beta']  # scale by regularization strength
        tf.summary.scalar("l2_penalty", l2_penalty_)
        regularized_loss_ = ce_loss_ + l2_penalty_

    # Sort out optimizer, add one from A4
    with tf.variable_scope("Training"):
        if params['optimizer'] == 'adagrad':
            optimizer_ = tf.train.AdagradOptimizer(params['lr'])
            train_op_ = optimizer_.minimize(regularized_loss_,
                                        global_step=tf.train.get_global_step())
        elif params['optimizer']=='adam':
            optimizer_ = tf.train.AdamOptimizer(learning_rate = params['lr'], name = "adam_optimizer")
            tvars = tf.trainable_variables()
            gradients = optimizer_.compute_gradients(regularized_loss_,tvars)
            grads = [x[0] for x in gradients]
            clipped, _ = tf.clip_by_global_norm(grads,params["maxGradNorm"])
            train_op_ = optimizer_.apply_gradients(zip(clipped,tvars))
        else:
            optimizer_ = tf.train.GradientDescentOptimizer(params['lr'])
            train_op_ = optimizer_.minimize(regularized_loss_,
                                            global_step=tf.train.get_global_step())
    
    tf.summary.scalar("cross_entropy_loss", ce_loss_)
    eval_metrics = {"cross_entropy_loss": tf.metrics.mean(ce_loss_),
                    "accuracy": tf.metrics.accuracy(labels, pred_max_)}
    
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions_dict,
                                      loss=regularized_loss_,
                                      train_op=train_op_,
                                      eval_metric_ops=eval_metrics)

# TODO: use the Estimator API as it was used in NeuralBOW.ipynb in A2, adjust parameters, train and test...
        
