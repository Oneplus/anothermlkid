#!/usr/bin/env python

import os
import sys

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from useless.instance import build_instance

try:
    from numpy import array, zeros, exp, log
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

def viterbi(model, instance):
    '''
    '''
    model.destroy_score_cache()
    build_instance(model.w, model.attrs, model.tags, instance, False)
    g0, g = model.build_score_cache(instance)

    L = len(instance)
    T = model.nr_tags

    s = zeros((L, T))
    p = zeros((L, T), dtype=int)

    s[0] = g0
    p[0] = array([-1] * T)

    for i in range(1, L):
        for t in range(T):
            s[i,t] = (s[i-1,] + g[i,:,t]).max()
            p[i,t] = (s[i-1,] + g[i,:,t]).argmax()

    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret

