#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

try:
    from numpy import exp, log
    from numpy.linalg import norm
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

def logsumexp(a):
    '''
    '''
    max_element = a.max()
    return max_element + log(exp(a - max_element).sum())
