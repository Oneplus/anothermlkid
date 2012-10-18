#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plot

def read_instances( fp, attrs, labels, append=False ):
    """
    read instances from file, the format of file is like this
    [class] [aid:value] ...
    
    @param  fp      the file handle
    @param  attrs   the attributes lexicon
    @param  labels  the labels lexicon
    @param  append  whether append the attribute and label to the lexicon
    @retunr X, Y    the instance vector and class vector
    """
    raw_data = []

    for line in fp:
        buff = line.strip().split()
        raw_data.append( (buff[0], [sep.split(":") for sep in buff[1:]]) )

    if append:
        for raw_inst in raw_data:
            y = int(raw_inst[0])
            if y not in labels:
                labels[y] = len(labels) + 1

            x = raw_inst[1]
            for xi in x:
                aid = int(xi[0])
                if aid not in attrs:
                    attrs[aid] = len(attrs) + 1

    L = len(attrs)

    X = []
    Y = []

    for raw_inst in raw_data:
        y = int(raw_inst[0])

        x = [0.] * L
        for xi in raw_inst[1]:
            aid, value = int(xi[0]), float(xi[1])
            x[aid - 1] = value

        if aid in attrs and y in labels:
            X.append( np.array(x) )
            Y.append( y )
        else:
            print >> sys.stderr, "Discard instance %s" % X

    return X, Y

def evaluate( X, Y, alpha, N, b, kernel, x):
    """
    @param  X       the instances
    @param  Y       the class
    @param  N       number of instances
    @param  b       
    @param  kernel  the kernel function
    @param  alpha   the alpha
    @param  x       the instance to be evaluate

    @return f(x)=w'x+b=\sum_j a_j*Y_j*<X_j,X_i>+b
    """

    return sum([alpha[j]*Y[j]*kernel(X[j], x) for j in xrange(N)])+b

def train( X, Y, tol, C, kernel ):
    """
    Train the SVM Model with tolerance and slack value C

    @param  X       the instances vector
    @param  Y       the class vector
    @param  tol     tolerance
    @param  C
    @param  kernel  the kernel function
    @return sv_x, sv_y, sv_alph support vector
    """
    b=.0
    N = len(X)
    alpha = np.array([0.] * N)

    def error(i):
        return evaluate( X, Y, alpha, N, b, kernel, X[i]) - Y[i]

    max_passes = 10
    max_iter   = 10000
    passes = 0
    it = 0
    while passes < max_passes and it < max_iter:
        changed = 0

        for i in xrange(N):
            Ei = error(i)

            # if this data point violate the KKT condition
            if ((Y[i]*Ei<-tol and alpha[i]<C) or
                    (Y[i]*Ei>tol and alpha[i]>0.)):
                j = i
                while j == i:
                    j = np.random.randint(0, N, size=1)

                Ej = error(j)

                L=0.; H=C;

                ai = alpha[i]
                aj = alpha[j]

                if Y[i] == Y[j]:
                    L=max(0., ai+aj-C)
                    H=min(C, ai+aj)
                else:
                    L=max(0., aj-ai)
                    H=min(C, C+aj-ai)

                if abs(L-H)<1e-4:
                    continue

                eta=2.*kernel(X[i],X[j])-kernel(X[i],X[i])-kernel(X[j],X[j])
                if eta >= 0.:
                    continue

                new_aj = aj - Y[j]*(Ei-Ej)/eta
                new_aj = H if new_aj>H else new_aj
                new_aj = L if new_aj<L else new_aj
                if abs(new_aj-aj)<1e-4:
                    continue

                alpha[j] = new_aj

                new_ai = ai + Y[i]*Y[j]*(aj-new_aj)
                alpha[i] = new_ai

                bi=(b-Ei-Y[i]*(new_ai-ai)*kernel(X[i],X[i])
                        -Y[j]*(new_aj-aj)*kernel(X[i],X[j]))
                bj=(b-Ej-Y[i]*(new_ai-ai)*kernel(X[i],X[j])
                        -Y[j]*(new_aj-aj)*kernel(X[j],X[j]))

                if new_ai>.0 and new_ai<C:
                    b=bi
                elif new_aj>.0 and new_aj<C:
                    b=bj
                else:
                    b=.5*(bi+bj)

                changed += 1

        it += 1

        if changed == 0:
            passes += 1
        else:
            passes = 0

    print >> sys.stderr, "Training is done."

    sv_x = []; sv_y = []; sv_alph = []
    for i in xrange(N):
        ev = evaluate(X, Y, alpha, N, b, kernel, X[i])*Y[i]
        if alpha[i] < C and alpha[i] > 0.:
            sv_x.append( X[i] )
            sv_y.append( Y[i] )
            sv_alph.append( alpha[i] )

    return (sv_x, sv_y, sv_alph, b)

def main():
    try:
        fp=open("data/simple.train.dat", "r")
    except:
        print >> sys.stderr, "Failed to open file"
        return

    attrs    = {}
    labels   = {}

    X, Y = read_instances( fp, attrs, labels, True )

    kernel = None

    # for test
    if True:
        kernel = lambda u,v: np.vdot(u, v)
    elif False:
        kernel = lambda u,v: math.exp(-np.vdot(u-v,u-v)/sigma)

    N = len(X)
    sv_x, sv_y, sv_alph, b = train( X, Y, 1e-4, 2., kernel )

    M = len(sv_x)

    if True:
        plot.xlim([-5.,5.])
        plot.ylim([-5.,5.])

        w = sum(np.transpose(
            np.array(sv_alph)
                *np.array(sv_y)
                *np.transpose(np.array(sv_x))))

        box = [-3.,-3.,3.,3.]

        def inbox(p):
            return (p[0]>=box[0] and
                    p[0]<=box[2] and
                    p[1]>=box[1] and
                    p[1]<=box[3])

        x00=box[0]; x01=(-b-w[0]*x00)/w[1]
        x10=box[2]; x11=(-b-w[0]*x10)/w[1]
        x21=box[1]; x20=(-b-w[1]*x21)/w[0]
        x31=box[3]; x30=(-b-w[1]*x31)/w[0]

        line1, line2, = [p for p in [(x00, x01),
            (x10,x11),
            (x20,x21),
            (x30,x31)] if inbox(p)]

        plot.plot([X[i][0] for i in xrange(N) if Y[i]==-1],
                [X[i][1] for i in xrange(N) if Y[i]==-1],
                'bx',
                [X[i][0] for i in xrange(N) if Y[i]==1],
                [X[i][1] for i in xrange(N) if Y[i]==1],
                'r+',
                [sv_x[i][0] for i in xrange(M) if sv_y[i]==-1],
                [sv_x[i][1] for i in xrange(M) if sv_y[i]==-1],
                'y^',
                [sv_x[i][0] for i in xrange(M) if sv_y[i]==1],
                [sv_x[i][1] for i in xrange(M) if sv_y[i]==1],
                'g^',
                [line1[0], line2[0]],
                [line1[1], line2[1]],
                'k-')

        plot.show()

    try:
        fp=open("data/simple.test.dat", "r")
    except:
        print >> sys.stderr, "Failed to open file"

    Xt, Yt = read_instances( fp, attrs, labels, False )

    ac = 0
    for i in xrange(len(Xt)):
        y = evaluate(sv_x, sv_y, sv_alph, len(sv_x), b, kernel, Xt[i])
        y = 1 if y > 0 else -1

        print "+1" if y > 0 else "-1"

        if y == Yt[i]:
            ac += 1

    print "accuracy=%.2lf%%" % (float(ac)/len(Xt)*100)

if __name__=="__main__":
    main()
