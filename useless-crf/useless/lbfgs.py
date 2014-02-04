#!/usr/bin/env python

import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

try:
    from scipy import optimize
except ImportError:
    print >> sys.stderr, "numpy is not installed"
    sys.exit(1)

from useless.logger import INFO, LOG
from useless.model import likelihood, dlikelihood

def lbfgs(model, instances):
    def callback(xk):
        LOG(INFO, "L-BFGS finish one iteration")

    model.w, f, d = optimize.fmin_l_bfgs_b(likelihood,
                                           model.w,
                                           fprime = dlikelihood,
                                           args = (instances, model),
                                           #iprint = 1,
                                           callback = callback)
