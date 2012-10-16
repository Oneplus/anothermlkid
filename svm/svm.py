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
    examine_all = 1

    def evaluate(i):
        pass

    def examine_example(i):
        assert (i >= 0 and i < N)
        y = Y[i]
        alphai = alpha[i]

        def getE(i):
            pass

        ei = getE(i)


    while num_changed > 0 or examine_all:
        num_changed = 0

        if examine_all:
            for i in xrange(N):
                num_changed += examine_example(i)


if __name__=="__main__":
    main()
