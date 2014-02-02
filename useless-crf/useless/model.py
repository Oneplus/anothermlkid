#!/usr/bin/env python
import sys
from collections import defaultdict

try:
    from numpy import array, zeros
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

class crfl2sgdmodel(object):

    DUMMY = '__x__'

    def __init__(self):
        self.tags = {self.DUMMY : 0}    # the index of tags
        self.attrs = {}                 # the index of attributes
        self.w = None                   # the parameters

        nr_tags  = -1
        nr_attrs = -1
        nr_dim   = -1

        _base = -1
        _eta = 0
        _lambda = 0

        self._g0 = None
        self._g  = None

    def info(self):
        return nr_tags, nr_attrs, nr_dim

    def destroy_score_cache(self):
        self._g0, self._g = None, None

    def build_score_cache(self, instance):
        '''
        Calculate the transition scoring matrix

        - @param[in] instance   the instance to be extracted
        '''

        if self._g0 is not None and self._g is not None:
            return (self._g0, self._g)

        L = len(instance.raw)
        T = self.nr_tags
        A = self.nr_attrs

        self._g0 = zeros(T, dtype=float)
        self._g = zeros((L, T, T), dtype=float)

        f = instance.features_table

        for j in xrange(T):
            self._g0[j] = self.w.take(f[0,None,j], axis=0).sum()

        for i in xrange(1, L):
            for k in xrange(T):
                self._g[i,0,k] = self.w.take(f[i,0,k], axis=0).sum()
                for j in xrange(1, T):
                   self._g[i,j,k] = self._g[i,0,k] - self.w[A*T+ k] + self.w[(A+j)*T + k]

        return (self._g0, self._g)

    def preprocess(self, instances):
        for instance in instances:
            for item in instance.raw:
                tag, attrs = item

                if tag not in self.tags:
                    self.tags[tag] = len(self.tags)

                for attr in attrs:
                    if attr not in self.attrs:
                        self.attrs[attr] = len(self.attrs)

        self.nr_tags = len(self.tags)
        self.nr_attrs = len(self.attrs)
        self.nr_dim = (self.nr_tags + self.nr_attrs) * self.nr_tags
        self.w = zeros(self.nr_dim, dtype=float)

    def build_instance(self, instance, train=True):
        instance.features_table = f = {}
        instance.correct_features = F = defaultdict(int)

        T = self.nr_tags
        A = self.nr_attrs


        previous_idx = None
        for i, item in enumerate(instance.raw):
            tag, attrs = item
            attrs = [self.attrs[attr] for attr in attrs if attr in self.attrs]
            if i == 0:
                for k in xrange(self.nr_tags):
                    f[i,None,k] = [attr * self.nr_tags + k for attr in attrs]
            else:
                for j in xrange(self.nr_tags):
                    for k in xrange(self.nr_tags):
                        f[i,j,k] = [attr * self.nr_tags + k for attr in attrs]
                        f[i,j,k].append((self.nr_attrs + j) * self.nr_tags + k)

            if train:
                idx = self.tags[tag]
                for attr in attrs:
                    F[attr * self.nr_tags + idx] += 1
                if i > 0:
                    F[(self.nr_attrs + previous_idx) * self.nr_tags + idx] += 1
                previous_idx = idx
