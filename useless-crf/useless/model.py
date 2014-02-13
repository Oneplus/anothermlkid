#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

import random
from collections import defaultdict

try:
    from numpy import array, zeros, exp, add, log
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

from useless.math       import logsumexp
from useless.logger     import INFO, WARN, ERROR, LOG, trace
from useless.instance   import build_instance, destroy_instance

DELTA = 2.

class crfmodel(object):
    '''
    The CRF model.
    '''

    DUMMY = '__x__'     # TAG for dummy tag

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


# A series HELP functions
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

from useless.viterbi    import forward, backward

def _likelihood(w, instance, model):
    '''
    Calculate the likelihood of one instance

    - param[in] w
    - param[in] Instance
    - param[in] model
    '''
    L = len(instance)
    T = model.nr_tags
    A = model.nr_attrs

    # Filling the correct_features and features_table
    build_instance(w, model.attrs, model.tags, instance, True)
    g0, g = build_score_cache(w, L, T, A, instance)

    # calcualte the correct likelihood
    F = instance.correct_features
    ret = array([w[k] * v for k, v in F.iteritems()]).sum()

    # calcualte the marginal
    a = forward(g0, g, L, T)
    destroy_instance(instance)

    return ret - logsumexp(a[L-1,:])

def likelihood(w, instances, model):
    '''
    Calculate the likelihood of the whole data set.

    - param[in] w           The weight vector {arraylike}
    - param[in] instances   The data set {useless.Instance}
    - param[in] model       The model
    '''
    N = len(instances)
    ret = 0.

    for index, instance in enumerate(instances):
        trace(index, N)
        ret += _likelihood(w, instance, model)

    return -(ret - ((w ** 2).sum() / 2 / (DELTA **2)))

def _dlikelihood(w, instance, model):
    '''
    Calculate gradient of a instance

    - param[in] w           The weight vector
    - param[in] instance    The instance
    - param[in] model       The model
    '''
    grad = zeros(w.shape[0], dtype=float)

    L = len(instance)
    T = model.nr_tags
    A = model.nr_attrs

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
    return grad


def dlikelihood(w, instances, model):
    '''
    Calculate the gradient on the overall data set.

    - param[w]  the
    '''
    N = len(instances)

    grad = zeros(w.shape[0], dtype=float)
    for index, instance in enumerate(instances):
        trace(index, N)
        grad += _dlikelihood(w, instance, model)

    return -(grad - w / DELTA)

def _gradient_test(w, instance, model, choosen_dims = None):
    '''
    The gradient test. Used to test if the gradient is correctly calculated.
    Detail of this method is described in Stochastic Gradient Descent Tricks
    by Bottou, L

    1. Pick an example z
    2. Compute the loss Q(z,w)
    3. Compute the gradient g = D_w Q(z,w)
    4. Apply a slightly pertubation to w'=w+\Delta.
    5. Compute the new loss Q(z,w') and verify Q(z,w')=Q(z,w)+g\Delta
    '''

    lossQ = _likelihood(w, instance, model)
    DlossQ = _dlikelihood(w, instance, model)

    build_instance(w, model.attrs, model.tags, instance, True)

    L = len(instance)
    T = len(model.tags)

    if not choosen_dims:
        f = instance.features_table
        features = []
        for i in range(L):
            for j in range(T):
                if i == 0:
                    features.extend(f[i,None,j])
                else:
                    for k in range(T):
                        features.extend(f[i,k,j])
        choosen_dims = random.sample(features, 5)

    epsilon = 1e-4
    grad_diff = epsilon * DlossQ[choosen_dims].sum()
    w[choosen_dims] += epsilon

    lossQ2 = _likelihood(w, instance, model)

    if abs(lossQ2 - (lossQ + grad_diff)) > 1e-7:
        LOG(WARN, "Failed gradient test.")
        LOG(WARN, "Pertubation on %s dims." % str(choosen_dims))
        LOG(WARN, "Loss before pertubation: %f" % lossQ)
        LOG(WARN, "Loss after pertubation: %f" % lossQ2)
        LOG(WARN, "Gradient difference: %f" % grad_diff)
    else:
        LOG(INFO, "Success gradient test.")

    w[choosen_dims] -= epsilon

def gradient_test(w, instances, model, choosen_dims = None):
    '''
    Sample some examples to perform gradient test
    '''
    for instance in random.sample(instances, min(5, len(instances))):
        _gradient_test(w, instance, model, choosen_dims)
