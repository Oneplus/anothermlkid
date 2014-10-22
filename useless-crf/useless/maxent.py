#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
if ROOTDIR not in sys.path:
    sys.path.append(ROOTDIR)

try:
    from numpy import exp, log
    import math
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

def logsumexp(a):
    '''
    Logsumexp
    ---------

    It is a little faster than the scipy.misc.logsumexp

    - param[in] array-like
    '''
    max_element = a.max()
    return max_element + math.log(exp(a - max_element).sum())
