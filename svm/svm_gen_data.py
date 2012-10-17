#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plot
import pylab

def svm_sample(N):
    return (np.array([np.random.normal(-1.5,1.,N),
                np.random.normal(-1.5,1.,N)]).reshape(N,2),
            np.array([np.random.normal(1.5,1.,N),
                np.random.normal(1.5,1.,N)]).reshape(N,2))

if __name__=="__main__":
    N = 20
    X1, X2 = svm_sample(N)

    plot.ylim([-5.,5.])
    plot.xlim([-5.,5.])
    plot.plot([X1[i, 0] for i in xrange(N)],
            [X1[i, 1] for i in xrange(N)],
            'rx',
            [X2[i, 0] for i in xrange(N)],
            [X2[i, 1] for i in xrange(N)],
            'b+')

    plot.savefig('simple_data.png')

    results = []

    for x in X1:
        results.append("+1\t1:%f\t2:%f" % (x[0], x[1]))

    for x in X2:
        results.append("-1\t1:%f\t2:%f" % (x[0], x[1]))

    try:
        fpo=open("data/simple.dat", "w")
    except:
        print >> sys.stderr, "Failed to open file"
        exit(1)

    np.random.shuffle(results)

    print >> fpo, "\n".join(results)
