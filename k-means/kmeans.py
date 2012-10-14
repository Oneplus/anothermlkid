#!/usr/bin/env python

import sys

from numpy import linalg, random, array, concatenate
import matplotlib.pyplot as plot

def sample(N):
    return (array([random.normal(-1.5,1.,N/2),random.normal(-1.5,1.,N/2)]).reshape(N/2, 2),
            array([random.normal(1.5,1.,N/2),random.normal(1.5,1.,N/2)]).reshape(N/2, 2))

def vertical(p1, p2, box):
    assert (box[0]<box[2] and box[1]<box[3])

    print p1, p2
    def inbox(p):
        return (p[0]>=box[0] and 
                p[0]<=box[2] and 
                p[1]>=box[1] and 
                p[1]<=box[3])

    k = (p2[1]-p1[1])/(p2[0]-p1[0])

    x0=box[0]; y0=-1./k*(x0-(p1[0]+p2[0])/2.)+((p1[1]+p2[1])/2.)
    x1=box[2]; y1=-1./k*(x1-(p1[0]+p2[0])/2.)+((p1[1]+p2[1])/2.)
    y2=box[1]; x2=-k*(y2-(p1[1]+p2[1])/2.)+((p1[0]+p2[0])/2.)
    y3=box[3]; x3=-k*(y3-(p1[1]+p2[1])/2.)+((p1[0]+p2[0])/2.)

    return [p for p in [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] if inbox(p)]

if __name__=="__main__":
    N = 1000
    X1, X2 = sample(N)
    X = concatenate([X1,X2])

    plot.plot([X1[i,0] for i in xrange(N/2)],
            [X1[i,1] for i in xrange(N/2)],
            'rx',
            [X2[i,0] for i in xrange(N/2)],
            [X2[i,1] for i in xrange(N/2)],
            'b+')

    plot.show()

    T = 10
    u1 = [array([random.rand()*2-2, random.rand()*2])]
    u2 = [array([random.rand()*2, random.rand()*2-2])]
    print u1
    print u2

    eps = 0.01

    Y = random.randint(2, size=N)
    #print Y
    for step in xrange(T):
        c1 = u1[step]
        c2 = u2[step]

        for i in xrange(N):
            x = X[i,]
            if linalg.norm(x - c1) < linalg.norm(x - c2):
                Y[i] = 0
            else:
                Y[i] = 1

        c1t = sum([X[i,] for i in xrange(N) if Y[i] == 0])/N
        c2t = sum([X[i,] for i in xrange(N) if Y[i] == 1])/N

        u1.append(c1t)
        u2.append(c2t)

        b1, b2, = vertical(c1t, c2t, [-3.,-3.,3.,3.])

        plot.plot([b1[0], b2[0]],
                [b1[1], b2[1]],
                'k-',
                [X[i,0] for i in xrange(N/2)],
                [X[i,1] for i in xrange(N/2)],
                'rx',
                [X[i,0] for i in xrange(N/2+1,N)],
                [X[i,1] for i in xrange(N/2+1,N)],
                'b+',
                c1t[0],
                c1t[1],
                'y^',
                c2t[0],
                c2t[1],
                'y^')

        print c1t, c2t
        plot.show()

        if linalg.norm(c1t - c1) < eps and linalg.norm(c2t - c2) < eps:
            print >> sys.stderr, "convergence"
            break

