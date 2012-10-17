#!/usr/bin/env python

import sys
import numpy as np

def main():
    try:
        fp=open("data/heart_scale.dat", "r")
    except:
        print >> sys.stderr, "Failed to open file"
        return

    attrs    = {}
    labels   = {}
    raw_data = []

    for line in fp:
        buff = line.strip().split()
        raw_data.append( (buff[0], [sep.split(":") for sep in buff[1:]]) )

    for raw_inst in raw_data:
        y = int(raw_inst[0])
        if y not in labels:
            labels[y] = len(labels) + 1

        x = raw_inst[1]
        for xi in x:
            aid = int(xi[0])
            if aid not in attrs:
                attrs[aid] = len(attrs) + 1

    L = (len(attrs) + 1)

    X = []
    Y = []

    for raw_inst in raw_data:
        y = int(raw_inst[0])

        x = [0.] * L
        for xi in raw_inst[1]:
            aid, value = int(xi[0]), float(xi[1])
            x[aid] = value

        X.append( np.array(x) )
        Y.append( y )

    N = len(X)

    alpha = np.array([0.] * N)

    num_changed = 0
    examine_all = True

    kernel = None

    if opt.kernel   == "LINEAR":
        kernel = lambda(u, v): np.vdot(u, v)
    elif opt.kerenl == "RBF":
        kernel = lambda(u,v): math.exp(-np.vdot(u-v,u-v)/sigma)

    def getE(i):
        return np.vdot(alpha, X[i])-Y[i]

    b=0.
    C=1.

    max_passes = 10
    max_iter   = 10000
    passes = 0
    it = 0
    while passes < max_passes and it < max_iter:
        changed = 0

        for i in xrange(N):
            Ei = getE(i)

            # if this data point violate the KKT condition
            if ((Y[i]*Ei<-tol and alpha[i]<C) or
                    (Y[i]*Ei>tol and alpha[i]>0.)):
                j = i
                while j == i:
                    j = np.random.randint(0, N, size=1)

                Ej = getE(j)

                L=0.; H=C;

                if Y[i] == Y[j]:
                    L=max(0., alpha[i]+alpha[j]-C)
                    H=min(C, alpha[i]+alpha[j])
                else:
                    L=max(0., alpha[j]-alpha[i])
                    H=min(C, C+alpha[j]-alpha[i])

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
                bj=(b-Ej-Y[j]*(new_ai-ai)*kernel(X[i],X[j])
                        -Y[j]*(new_aj-aj)*kernel(X[j],X[j]))

                b=.5*(bi+bj)

                if new_ai>.0 and new_ai<C:
                    b=bi
                if new_aj>.0 and new_aj<C:
                    b=bj

                chanaged += 1

        it += 1

        if changed == 0:
            passes += 1
        else:
            passes = 0



if __name__=="__main__":
    main()
