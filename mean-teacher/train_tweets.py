# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Train Feedforward NN Mean Teacher on combined (labeled + unlabeled) tweet training set and evaluate against a validation set

"""

import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets.tweets import TweetData
from mean_teacher import minibatching
from mean_teacher.w266_model import W266Model


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=42):
    n_extra_unlabeled = 20000

    model = W266Model(RunContext(__file__, 0)) 

    tensorboard_dir = model.save_tensorboard_graph() 
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    tweetData = TweetData(data_seed, n_extra_unlabeled)
    training_batches = minibatching.training_batches(tweetData.training, n_labeled_per_batch='vary')
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(tweetData.evaluation)

    model.train(training_batches, evaluation_batches_fn)
    
    LOG.info("Result on test set:")
    tweetDataTest = TweetData(data_seed, test_phase=True);
    evaluation_test_batches_fn = minibatching.evaluation_epoch_generator(tweetDataTest.evaluation)
    model.evaluate(evaluation_test_batches_fn)
    
if __name__ == "__main__":
    run()
