#!/usr/bin/env python

import os
import sys

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from useless.maxent     import logsumexp
from useless.model      import build_score_cache
from useless.instance   import build_instance, destroy_instance

try:
    from numpy import array, zeros, exp, log
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)


def forward(g0, g, L, T):
    '''
    The forward process of forward-backward algorithm.

    Parameters
    ----------
    g0 : array of T
        $\Phi(0,None,k)$
    g : array of L*T*T
        $\Phi(i,j,k)$
    L : int
        length of instance
    T : int
        number of tags

    Returns
    -------
    a : array of L*T
        The forward process matrix
    '''
    a = zeros((L, T), dtype=float)
    a[0,:] = g0
    for i in xrange(1, L):
        ap = a[i-1,:]
        for o in xrange(T):
            a[i,o] = logsumexp(ap + g[i,:,o])
    return a


def backward(g, L, T):
    '''
    The backward process of forward-backward algorithm.

    Parameters
    ----------
    g : array of L*T*T
        $\Phi(i,j,k)$
    L : int
        length of instance
    T : int
        number of tags

    Returns
    -------
    b : array of L*T
        The forward process matrix
    '''
    b = zeros((L, T), dtype=float)
    for i in xrange(L-2, -1, -1):
        bp = b[i+1,:]
        for o in xrange(T):
            b[i,o] = logsumexp(bp + g[i+1,o,:])
    return b


def argmax(g0, g, L, T):
    '''
    argmax
    ------

    - param[in] g0
    - param[in] g
    - param[in] L
    - param[in] T
    '''
    s = zeros((L, T), dtype=float)
    p = zeros((L, T), dtype=int)

    s[0] = g0
    p[0] = array([-1] * T)

    for i in range(1, L):
        for t in range(T):
            s[i,t] = (s[i-1,] + g[i,:,t]).max()
            p[i,t] = (s[i-1,] + g[i,:,t]).argmax()

    return s, p


def viterbi(model, instance):
    '''
    '''
    L = len(instance)
    T = model.nr_tags
    A = model.nr_attrs

    build_instance(model.attrs, model.tags, instance, False)
    g0, g = build_score_cache(model.w, L, T, A, instance)
    destroy_instance(instance)

    s, p = argmax(g0, g, L, T)

    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret

