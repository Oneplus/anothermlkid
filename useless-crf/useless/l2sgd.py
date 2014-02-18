#!/usr/bin/env python

import sys
import os
import random

from collections import defaultdict

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from useless.maxent     import logsumexp
from useless.logger     import INFO, WARN, ERROR, LOG
from useless.model      import build_score_cache
from useless.viterbi    import forward, backward
from useless.instance   import build_instance, destroy_instance

try:
    from numpy import array, zeros, exp, log, sqrt, add
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

def expectation(model, instance):
    '''
    Perform forward-backward algorithm to calculate the second component
    of the detrieve.
    '''
    # get the cached score

    L = len(instance)
    T = model.nr_tags
    A = model.nr_attrs
    g0, g = build_score_cache(model.w, L, T, A, instance)

    a = forward(g0, g, L, T)
    b = backward(g, L, T)

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


def tune_eta0(samples):
    LOG(INFO, "Tuning eta0 from %d samples" % len(samples))

def l2sgd(model,
          instances,
          nr_epoth,
          init_learning_rate,
          adjust_learning_rate = False):

    _sigma = 1.
    _gamma = init_learning_rate
    _t = 1.

    _eta = 0.
    samples = random.sample(instances, min(int(len(instances) * 0.1), 1000))

    for epoth in xrange(nr_epoth):
        LOG(INFO, "Training epoth [%d]" % epoth)
        # randomly shuffle the training instances
        random.shuffle(instances)

        # loop over the training instances
        for index, instance in enumerate(instances):
            # first need to clear the cache
            build_instance(model.w, model.attrs, model.tags, instance)

            for k, v in expectation(model, instance).iteritems():
                model.w[k] -= v * _gamma
            for k, v in instance.correct_features.iteritems():
                model.w[k] += v * _gamma

            # re-calculate the scale
            _gamma = init_learning_rate / (1 + sqrt(float(epoth)))

            _t += 1.

            if (index + 1) % 1000 == 0:
                LOG(INFO, "%d instances is trained" % (index + 1))

            destroy_instance(instance)

        LOG(INFO, "%d instances is trained" % (index + 1))
        LOG(INFO, "Parameters norm %f" % norm(model.w))
