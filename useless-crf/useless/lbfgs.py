#!/usr/bin/env python

import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

try:
    from scipy import optimize
except ImportError:
    print >> sys.stderr, "scipy is not installed"
    sys.exit(1)

from useless.logger import INFO, LOG
from useless.model import likelihood, dlikelihood, gradient_test

def lbfgs(model, instances):
    '''
    Invoke the scipy.optimize.fmin_l_bfgs_b to perform optimization
    '''
    def callback(xk):
        LOG(INFO, "L-BFGS finish one iteration")
        LOG(INFO, "Gradient test starts: ")
        gradient_test(model.w, instances, model)

    model.w, f, d = optimize.fmin_l_bfgs_b(likelihood,
                                           model.w,
                                           fprime = dlikelihood,
                                           args = (instances, model),
                                           iprint = 1,
                                           factr = 1e12,
                                           callback = callback)
