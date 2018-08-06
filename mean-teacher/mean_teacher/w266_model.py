# file where our model of combined mean-teacher and feedforward nn will live.
"Mean teacher model for W266 project, heavily adapted from https://github.com/CuriousAI/mean-teacher/blob/master/tensorflow/mean_teacher/model.py and other files in the mean-teacher repo."

import logging
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean
from . import w266_nn as nn
from . import w266_wn as wn
from .framework import ema_variable_scope, name_variable_scope, assert_shape, HyperparamVariables
from . import string_utils


LOG = logging.getLogger('main')

class W266Model:
    DEFAULT_HYPERPARAMS = {
        # Consistency hyperparameters
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 50.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'num_logits': 1, # Either 1 or 2

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 5000,
        'rampdown_length': 8000,
        'training_length': 10000,

        # Output schedule
        'print_span': 125,
        'evaluation_span': 500,

        # list of hidden layers and their size
        'hidden_dims':[300, 300, 300, 300,300,300]
    }
    #pylint: disable=too-many-instance-attributes
    def __init__(self, run_context=None):
        self.name = "Tweet Data Class"
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.tweets = tf.placeholder(dtype=tf.float32, shape=(None, 500), name='tweets')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)
        
        with tf.name_scope("ramps"):
            # Ramp-up and ramp-down has been removed for simplicity
#             self.learning_rate = tf.constant(self.hyper['max_learning_rate'], dtype = tf.float32)
#             self.adam_beta_1 = tf.constant(self.hyper['adam_beta_1_after_rampdown'], dtype = tf.float32)
#             self.cons_coefficient = tf.constant(self.hyper['max_consistency_cost'], dtype = tf.float32)
#             self.adam_beta_2 = tf.constant(self.hyper['adam_beta_2_after_rampup'], dtype = tf.float32)
#             self.ema_decay = tf.constant(self.hyper['ema_decay_after_rampup'], dtype = tf.float32)
#             self.learning_rate =self.DEFAULT_HYPERPARAMS['max_learning_rate']
#             self.adam_beta_1 = self.DEFAULT_HYPERPARAMS['adam_beta_1_after_rampdown']
#             self.cons_coefficient = self.DEFAULT_HYPERPARAMS['max_consistency_cost']
#             self.adam_beta_2 = self.DEFAULT_HYPERPARAMS['adam_beta_2_after_rampup']
#             self.ema_decay = self.DEFAULT_HYPERPARAMS['ema_decay_after_rampup']
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')
            
        
        # below is where the interesting stuff happens, mostly.
        # Inference is a function which creates the towers and sets up the different logits for the two models
        (
            (self.class_logits_1, self.cons_logits_1),
            (self.class_logits_2, self.cons_logits_2),
            (self.class_logits_ema, self.cons_logits_ema)
        ) = inference(
            self.tweets,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            hidden_dims = self.DEFAULT_HYPERPARAMS['hidden_dims'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            num_logits=self.hyper['num_logits'])
        
        with tf.name_scope("objectives"):
            # something weird is done with errors for unlabeled examples. 
            # I think errors are only calculated for labeled, but you don't calculate it for unlabeled, so it is NaN for unlabeled
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels)
            # where we calculate classification costs.
            # the cost_1 should be for student and ema is for teacher
            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs( self.cons_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask)


            def l2_norms(matrix):
                l2s = tf.reduce_sum(matrix ** 2, axis=1)
                mean_l2 = tf.reduce_mean(l2s)
                return mean_l2, l2s

            self.mean_res_l2_1, self.res_l2s_1 = l2_norms(self.class_logits_1 - self.cons_logits_1)
            self.mean_res_l2_ema, self.res_l2s_ema = l2_norms(self.class_logits_ema - self.cons_logits_ema)

            # mean total cost is what you are optimizng. 
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt)
            assert_shape(self.total_costs_mt, [2])

            self.cost_to_be_minimized = self.mean_total_cost_mt

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_op = adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])

        # TODO do we really need this?
        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        self.training_metrics = {
            # NOTE these should not need training, since we don't do ramp-up and ramp-down
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        # TODO not sure what streaming mean does?
        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        # TODO string utils just formats dictionary results in a nice way for logging, not needed?
        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            print("Train init variables:")
            for var in train_init_variables:
                print(var)
            self.train_init_op = tf.variables_initializer(train_init_variables)
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.run(self.init_init_op)
    
    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)
    
    # TODO is this ok, do we understand it?
    def train(self, training_batches, evaluation_batches_fn):
#         self.run(tf.global_variables_initializer())
#         print("GLOBAL VAR INITI RAN")
#         self.run(tf.local_variables_initializer())
#         print("LOCAL VAR INITI RAN")
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch))
            step_control = self.get_training_control()
            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    # TODO is this ok, do we understand it?
    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)
    
    def feed_dict(self, batch, is_training=True):
#         print(type(batch))
#         print(batch.shape)
#         print(batch['x'].shape)
#         print(batch['y'].shape)
        return {
            self.tweets: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()

def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")

# understand this well
def inference(inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
              num_logits, hidden_dims):
    tower_args = dict(inputs=inputs,
                      is_training=is_training,
                      input_noise=input_noise,
                      num_logits=num_logits,
                      hidden_dims=hidden_dims)

    with tf.variable_scope("initialization") as var_scope:
        _ = tower(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        # This is the student model
        class_logits_1, cons_logits_1 = tower(**tower_args, dropout_probability=student_dropout_probability, name=name_scope)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        # This is the teacher model, but we train it normally; This isn't used
        class_logits_2, cons_logits_2 = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        class_logits_ema, cons_logits_ema = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
        # NOTE tf.stop_gradient just stops the gradient from updating these parameters, it just leaves the output as the input
        # This is definitely for the teacher model, which we use in the end for consistency
        class_logits_ema, cons_logits_ema = tf.stop_gradient(class_logits_ema), tf.stop_gradient(cons_logits_ema)
    return (class_logits_1, cons_logits_1), (class_logits_2, cons_logits_2), (class_logits_ema, cons_logits_ema)
    
# TODO create tower
def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          num_logits,
          hidden_dims,
          is_initialization=False,
          name=None):
    with tf.name_scope(name, "tower"):
        training_mode_funcs = [
            nn.gaussian_noise,wn.fully_connected, wn.affine_layer
        ]
        training_args = dict(
            is_training=is_training
        )

        with slim.arg_scope(training_mode_funcs, **training_args):

            noisy_inputs = nn.gaussian_noise(inputs, scale = input_noise, name = 'gaussian_noise')

            net = noisy_inputs
            net = wn.affine_layer(net, hidden_dims[0], activation_fn = lrelu, init = is_initialization)
            net = wn.affine_layer(net, hidden_dims[1], activation_fn = lrelu, init = is_initialization)
            net = slim.dropout(net, 1-dropout_probability, scope = "dropout_probability_1")
            net = wn.affine_layer(net, hidden_dims[2], activation_fn = lrelu, init = is_initialization)
            net = wn.affine_layer(net, hidden_dims[3], activation_fn = lrelu, init = is_initialization)
            net = slim.dropout(net, 1-dropout_probability, scope = "dropout_probability_2")
            # NOTE expand below in accordance with hidden_dims size and how many affine layers you want
#             net = wn.affine_layer(net, hidden_dims[4], activation_fn = lrelu, init = is_initialization)
#             net = wn.affine_layer(net, hidden_dims[5], activation_fn = lrelu, init = is_initialization)
#             net = slim.dropout(net, 1-dropout_probability, scope = "dropout_probability_3")
            
            primary_logits = wn.fully_connected(net, 2, init=is_initialization)
            secondary_logits = wn.fully_connected(net, 2, init=is_initialization)
            with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                            tf.assert_less_equal(num_logits, 2)]):
                secondary_logits = tf.case([
                    (tf.equal(num_logits, 1), lambda: primary_logits),
                    (tf.equal(num_logits, 2), lambda: secondary_logits),
                ], exclusive=True, default=lambda: primary_logits)

            assert_shape(primary_logits, [None, 2])
            assert_shape(secondary_logits, [None, 2])
            return primary_logits, secondary_logits
        

def lrelu(inputs, leak=0.1, name=None):
    with tf.name_scope(name, 'lrelu') as scope:
        return tf.maximum(inputs, leak * inputs, name=scope)

def adam_optimizer(cost, global_step,
                   learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                   name=None):
    with tf.name_scope(name, "adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        return optimizer.minimize(cost, global_step=global_step, name=scope)

def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }

def errors(logits, labels, name=None):
    """Compute error mean and whether each labeled? example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over labeled? examples.
    Mean error is NaN if there are no labeled? examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, name=None):
    """Takes a softmax of the logits and returns their distance as described below
       Use MSE as distance metric
    """

    with tf.name_scope(name, "consistency_costs") as scope:
        num_classes = 2
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1) * tf.to_float(mask) * cons_coefficient
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs, axis=1)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs
    
