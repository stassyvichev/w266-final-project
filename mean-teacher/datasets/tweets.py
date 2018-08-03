# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
from .utils import random_partitions

class Datafile:
    def __init__(self, path):
        self.path = path
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data

    def _load(self):
        self._data = np.load(self.path);

class TweetData:
    

    def __init__(self,data_seed=42, n_extra_unlabeled=0, test_phase=False):
        random = np.random.RandomState(seed=data_seed)
        DIR = os.path.join('..','data', 'data_mean_teacher')
        self.FILES = {
            'dev': Datafile(os.path.join(DIR, 'encoded_labeled_dev_data.npy')),
            'train': Datafile(os.path.join(DIR, 'encoded_labeled_train_data.npy')),
            'extra': Datafile(os.path.join(DIR, 'encoded_unlabeled_data.npy')),
            'test': Datafile(os.path.join(DIR, 'encoded_labeled_test_data.npy')),
        }
        print(self.FILES['dev'])
        
        if test_phase:
            self.evaluation, self.training = self.FILES['dev'].data, self.FILES['train'].data
        else:
            self.evaluation, self.training = self.FILES['test'].data, self.FILES['train'].data

        if n_extra_unlabeled > 0:
            self.training = self._add_extra_unlabeled(self.training, n_extra_unlabeled, random)

    def _add_extra_unlabeled(self, data, n_extra_unlabeled, random):
        extra_unlabeled, _ = random_partitions(self.FILES['extra'].data, n_extra_unlabeled, random)
        return np.concatenate([data, extra_unlabeled])
