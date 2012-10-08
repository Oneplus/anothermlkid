#!/usr/bin/env python

from numpy import random, array, concatenate
import matplotlib.pyplot as plot

def sample(N):
    X = concatenate([random.normal(-1.5, 1., N),
        random.normal(1.5, 1., N)])
    Y = concatenate([random.normal(-1.5, 1., N),
        random.normal(1.5, 1., N)])

    return [X, Y]

if __name__=="__main__":
    N = 1000
    X1, X2 = sample(N)

    #plot.plot(X, Y, 'r.')
    #plot.show()

    M = 100
    u1 = (random.rand()*4-2, random.rand()*4-2)
    u2 = (random.rand()*4-2, random.rand()*4-2)
    print u1, u2
    
    for step in xrange(M):
        pass
