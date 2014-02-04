#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from collections import defaultdict

try:
    from numpy import array, zeros, exp, add, log
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

class crfl2sgdmodel(object):

    DUMMY = '__x__'

    def __init__(self):
        self.tags = {self.DUMMY : 0}    # the index of tags
        self.attrs = {}                 # the index of attributes
        self.w = None                   # the parameters

        self._g0 = None
        self._g  = None

    def info(self):
        return nr_tags, nr_attrs, nr_dim

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

def build_score_cache(w, L, T, A, instance):
    '''
    Calculate the transition scoring matrix

    - @param[in]    w           the parameter
    - @param[in]    L           the length of instance
    - @param[in]    T           the number of tags
    - @param[in]    A           the number of attributes
    - @param[in]    instance    the instance to be extracted
    '''
    g0 = zeros(T, dtype=float)
    g = zeros((L, T, T), dtype=float)

    f = instance.features_table

    for j in xrange(T):
        g0[j] = w.take(f[0,None,j], axis=0).sum()

    for i in xrange(1, L):
        for k in xrange(T):
            g[i,0,k] = w.take(f[i,0,k], axis=0).sum()
            for j in xrange(1, T):
               g[i,j,k] = g[i,0,k] - w[A*T+ k] + w[(A+j)*T + k]

    return (g0, g)

from useless.math       import logsumexp
from useless.logger     import INFO, LOG, trace
from useless.model      import build_score_cache
from useless.viterbi    import forward, backward
from useless.instance   import build_instance, destroy_instance

#@profile
def likelihood(w, instances, model):
    T = model.nr_tags
    A = model.nr_attrs
    N = len(instances)
    ret = 0.

    for index, instance in enumerate(instances):
        trace(index, N)

        L = len(instance)

        # Filling the correct_features and features_table
        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = build_score_cache(w, L, T, A, instance)

        # calcualte the correct likelihood

        F = instance.correct_features
        ret += array([w[k] * v for k, v in F.iteritems()]).sum()

        # calcualte the marginal
        a = forward(g0, g, L, T)

        ret -= logsumexp(a[L-1,:])

        destroy_instance(instance)

    #print >> sys.stderr, ret
    return -(ret - ((w ** 2).sum() / 2))

#@profile
def dlikelihood(w, instances, model):
    N = len(instances)
    T = model.nr_tags
    A = model.nr_attrs

    grad = zeros(w.shape[0], dtype=float)
    for index, instance in enumerate(instances):
        trace(index, N)

        L = len(instance)
        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = build_score_cache(w, L, T, A, instance)

        F = instance.correct_features
        for k, v in F.iteritems():
            grad[k] += v

        a = forward(g0, g, L, T)    # forward
        b = backward(g, L, T)       # backward

        logZ = logsumexp(a[L-1,:])

        f = instance.features_table

        c = exp(g0 + b[0,:] - logZ).clip(0., 1.)
        for j in xrange(T):
            grad[f[0,None,j]] -= c[j]

        for i in xrange(1, L):
            c = exp(add.outer(a[i-1,:], b[i,:]) + g[i,:,:] - logZ).clip(0.,1.)
            for j in range(T):
                for k in range(T):
                    grad[f[i,j,k]] -= c[j,k]

        destroy_instance(instance)

    return -(grad - w)

