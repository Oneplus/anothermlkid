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
from useless.instance import build_instance, destroy_instance

#@profile
def likelihood(w, instances, model):
    T = model.nr_tags
    ret = 0.
    for index, instance in enumerate(instances):
        L = len(instance)

        # Filling the correct_features and features_table
        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = model.build_score_cache(instance)

        # calcualte the correct likelihood

        F = instance.correct_features
        ret += array([w[k] * v for k, v in F.iteritems()]).sum()

        # calcualte the marginal
        a = zeros((L, T), dtype=float)
        a[0,:] = g0
        for i in xrange(1, L):
            for o in xrange(T):
                a[i,o] = logsumexp(a[i-1,:] + g[i,:,o])

        logZ = logsumexp(a[L-1,:])

        ret -= logZ

        model.destroy_score_cache()
        destroy_instance(instance)

    ret -= norm(w)
    return ret

#@profile
def dlikelihood(w, instances, model):
    grad = zeros(w.shape[0])

    for instance in instances:

        L = len(instance)
        T = model.nr_tags

        build_instance(w, model.attrs, model.tags, instance, True)
        g0, g = model.build_score_cache(instance)

        # forward
        # - exp{a[i,j]} = \sum k exp{a[i-1,k] + U[i,j] + B[k,j]}
        # - a[i,j] = log \sum k exp{a[i-1,k] + U[i,j] + B[k,j]}
        #          = U[i,j] + log \sum k exp{a[i-1,k] + B[k,j]}

        F = instance.correct_features
        for k, v in F.iteritems():
            grad[k] += v

        # forward
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

    grad -= w
    return grad

def lbfgs(model, instances):

    like = lambda x : -likelihood(x, instances, model)
    dlike = lambda x : -dlikelihood(x, instances, model)

    def callback(xk):
        print >> sys.stderr, "TRACE : lbfgs training 1 iter done."

    model.w, f, d = optimize.fmin_l_bfgs_b(like,
                                           model.w,
                                           fprime = dlike,
                                           maxiter = 10,
                                           callback = callback)
