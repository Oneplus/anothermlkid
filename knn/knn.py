#!/usr/bin/env python

from numpy import linalg, random, array, concatenate
import matplotlib.pyplot as plot

def sample(N):
    X = concatenate([random.normal(-1.5, 1., N/2),
        random.normal(1.5, 1., N/2)])
    Y = concatenate([random.normal(-1.5, 1., N/2),
        random.normal(1.5, 1., N/2)])

    return array([X, Y]).reshape(N, 2)

if __name__=="__main__":
    N = 1000
    X = sample(N)

    print X
    #plot.plot(X, Y, 'r.')
    #plot.show()

    T = 10
    u1 = [array([random.rand()*4-2, random.rand()*4-2])]
    u2 = [array([random.rand()*4-2, random.rand()*4-2])]
    print u1
    print u2

    Y = random.randint(2, size=N)
    #print Y
    for step in xrange(T):
        c1 = u1[step]
        c2 = u2[step]

        for i in xrange(N):
            X = X[i]
            if linalg.norm(X - c1) < linalg.norm(X - c2):
                Y[i] = 0
            else:
                Y[i] = 1

        c1t = sum([X1[i] for i in xrange(N) if Y[i] == 0])/N
        c2t = sum([X2[i] for i in xrange(N) if Y[i] == 1])/N

        u1.append(c1t)
        u2.append(c2t)

        print c1t, c2t

