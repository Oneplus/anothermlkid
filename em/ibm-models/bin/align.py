#!/usr/bin/env python
import sys
import os
import random

from collections import defaultdict

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
if ROOTDIR not in sys.path:
    sys.path.append(ROOTDIR)

from ibm.model1 import Model1
from ibm.model2 import Model2

if __name__=="__main__":
    from optparse import OptionParser

    usage = "%s --english=<target-file> --french=<source-file> [options]" % sys.argv[0]
    parser = OptionParser(usage)
    parser.add_option("--english",
                      dest="english", default="data/hansards.e",
                      help="The English file")
    parser.add_option("--french",
                      dest="french", default="data/hansards.f",
                      help="The French file")
    parser.add_option("--m1-iter",
                      dest="m1_max_iter", type=int, default=5,
                      help="The max number of iteration in M1")
    parser.add_option("--m2-iter",
                      dest="m2_max_iter", type=int, default=5,
                      help="The max number of iteration in M2")
    parser.add_option("--n",
                      dest="n", default=100,
                      help="The number of instances")
    parser.add_option("--output",
                      dest="prefix", default="ibm",
                      help="The output prefix")

    opts, args = parser.parse_args()

    model1 = Model1(opts.english, opts.french, opts.n, opts.m1_max_iter)
    model1.em()
    model1.save(opts.output + ".m1.pkl")

    model2 = Model2(opts.english, opts.french, opts.m2_max_iter)
    model2.em(opts.output + ".m1.pkl")
