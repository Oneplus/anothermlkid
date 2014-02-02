#!/usr/bin/env python

import sys
import os
import random

from collections import defaultdict

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

try:
    from numpy import array, zeros, exp, log, sqrt, add
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

def logsumexp(a):
    '''
    '''
    max_element = a.max()
    return max_element + log(exp(a - max_element).sum())

#@profile
def expectation(model, instance):
    '''
    Perform forward-backward algorithm to calculate the second component
    of the detrieve.
    '''
    # get the cached score
    g0, g = model.build_score_cache(instance)

    L = len(instance)
    T = model.nr_tags

    # forward
    # - exp{a[i,j]} = \sum k exp{a[i-1,k] + U[i,j] + B[k,j]}
    # - a[i,j] = log \sum k exp{a[i-1,k] + U[i,j] + B[k,j]}
    #          = U[i,j] + log \sum k exp{a[i-1,k] + B[k,j]}
    a = zeros((L, T), dtype=float)
    a[0,:] = g0
    for i in xrange(1, L):
        for o in xrange(T):
            a[i,o] = logsumexp(a[i-1,:] + g[i,:,o])

    # backward
    b = zeros((L, T), dtype=float)
    for i in xrange(L-2, -1, -1):
        for o in xrange(T):
            b[i,o] = logsumexp(b[i+1,:] + g[i+1,o,:])

    logZ = logsumexp(a[L-1,:])

    E = defaultdict(float)
    f = instance.features_table

    c = exp(g0 + b[0,:] - logZ).clip(0., 1.)
    for j in xrange(T):
        for k in f[0,None,j]:
            E[k] += c[j]

    for i in xrange(1, L):
        c = exp(add.outer(a[i-1,:], b[i,:]) + g[i,:,:] - logZ).clip(0.,1.)
        for j in range(T):
            for k in range(T):
                for e in f[i,j,k]:
                    E[e] += c[j,k]

    return E

def L2SGD(model,
          instances,
          nr_epoth,
          init_learning_rate,
          adjust_learning_rate = False):

    _lambda = 1.
    _gamma = init_learning_rate
    _t = 1.

    for epoth in xrange(nr_epoth):
        print >> sys.stderr, "TRACE : Training epoth [%d]" % epoth
        # !NOTE randomly shuffle the training instances
        random.shuffle(instances)

        # loop over the training instances
        for index, instance in enumerate(instances):
            # first need to clear the cache
            model.destroy_score_cache()
            model.build_instance(instance)

            for k, v in expectation(model, instance).iteritems():
                model.w[k] -= v * _gamma
            for k, v in instance.correct_features.iteritems():
                model.w[k] += v * _gamma

            # re-calculate the scale
            _gamma = init_learning_rate / (1 + sqrt(float(epoth)))

            _t += 1.

            if index % 1000 == 0:
                print >> sys.stderr, "TRACE : %d instances is trained" % index

            instance.features_table = None
            instance.correct_features = None

        print >> sys.stderr, "TRACE : Parameters norm %f" % norm(model.w)
