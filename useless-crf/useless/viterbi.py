#!/usr/bin/env python

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
    model.build_instance(instance, False)
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

