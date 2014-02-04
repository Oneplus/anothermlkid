#!/usr/bin/env python

import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

try:
    from numpy import array, zeros, log, exp, add
    from numpy.linalg import norm
    from scipy import optimize
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

from useless.math import logsumexp
from useless.viterbi import forward, backward
from useless.instance import build_instance, destroy_instance

def trace(index):
    if (index + 1) % 100 == 0:
        print >> sys.stderr, "%4d" % (index + 1),
        if (index + 1) % 1000 == 0:
            print >> sys.stderr

#@profile
def likelihood(w, instances, model):
    T = model.nr_tags
    ret = 0.

    for index, instance in enumerate(instances):
        trace(index)

        L = len(instance)

        # Filling the correct_features and features_table
        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = model.build_score_cache(instance)

        # calcualte the correct likelihood

        F = instance.correct_features
        ret += array([w[k] * v for k, v in F.iteritems()]).sum()

        # calcualte the marginal
        a = forward(g0, g, L, T)

        ret -= logsumexp(a[L-1,:])

        model.destroy_score_cache()
        destroy_instance(instance)

    #print >> sys.stderr, ret
    return -(ret - ((w ** 2).sum() / 2))


#@profile
def dlikelihood(w, instances, model):
    T = model.nr_tags

    grad = zeros(w.shape[0], dtype=float)
    for index, instance in enumerate(instances):
        trace(index)

        L = len(instance)
        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = model.build_score_cache(instance)

        F = instance.correct_features
        for k, v in F.iteritems():
            grad[k] += v

        # forward
        a = forward(g0, g, L, T)
        # backward
        b = backward(g, L, T)

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

        model.destroy_score_cache()
        destroy_instance(instance)

    return -(grad - w)

def lbfgs(model, instances):
    def callback(xk):
        print >> sys.stderr, "TRACE : lbfgs training 1 iter done."

    model.w, f, d = optimize.fmin_l_bfgs_b(likelihood,
                                           model.w,
                                           fprime = dlikelihood,
                                           args = (instances, model),
                                           #iprint = 1,
                                           callback = callback)
