#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from useless.model import crfl2sgdmodel
from useless.l2sgd import L2SGD
from useless.viterbi import viterbi
from useless.instance import Instance

def evaluate(model, eval_file):
    fp = open(eval_file, "r")
    instances = [Instance(c) for c in fp.read().strip().split("\n\n")]

    model.preprocess(instances, False)

    nr_correct = 0
    nr_tags = 0
    for instance in instances:
        predict = viterbi(model, instance)
        for index, word in enumerate(instance.raw):
            if predict[index] == model.tags.get(word[0], 0):
                nr_correct += 1

        nr_tags += len(instance)

    acc = float(nr_correct) / nr_tags
    print >> sys.stderr, "TRACE : accuracy = %f (%d/%d)" % (acc, nr_correct, nr_tags)


def learn(opts):
    try:
        fp = open(opts.train_file, "r")
    except IOError:
        print >> sys.stderr, "ERROR: Failed to open train file %s:" % opts.train_file
        return

    m = crfl2sgdmodel()
    instances = [Instance(c) for c in fp.read().strip().split("\n\n")]
    fp.close()

    m.preprocess(instances)

    print >> sys.stderr, "TRACE : number of tags %d" % m.nr_tags
    print >> sys.stderr, "TRACE : number of attributes %d" % m.nr_attrs
    print >> sys.stderr, "TRACE : number of instances %d" % len(instances)
    print >> sys.stderr, "TRACE : paramter dimision is %d" % m.nr_dim

    for i in xrange(opts.epoth):
        if opts.algorithm == "l2sgd":
            L2SGD(model              = m,
                  instances          = instances,
                  nr_epoth           = opts.cepoth,
                  init_learning_rate = opts.eta)

        elif opts.algorithm == "lbfgs":
            lbfgs(instances)

        if opts.dev_file:
            evaluate(m, opts.dev_file)

if __name__=="__main__":
    from optparse import OptionParser

    usage = "This is a toy program of linear-chained CRF"
    optparser = OptionParser(usage)
    optparser.add_option("-t", "--train", dest="train_file",
                         help="specify training file")
    optparser.add_option("-d", "--dev",  dest="dev_file",
                         help="specify development file")
    optparser.add_option("-a", "--algorithm", dest="algorithm",
                         help="specify training algorithm",
                         default="l2sgd")
    optparser.add_option("-e", "--eta", dest="eta",
                         help="initial learning rate",
                         type = float, default = 1.)
    optparser.add_option("-i", "--epoth", dest="epoth",
                         help="specify training epoth",
                         type = int, default = 10)
    optparser.add_option("-c", "--cepoth", dest="cepoth",
                         help="specify training interval length",
                         type = int, default = 3)

    opts, args = optparser.parse_args()

    if len(args) == 0 or args[0] not in ["learn", "tag"]:
        print >> sys.stderr, "ERROR: command [learn/tag] must be specified.\n"
        optparser.print_help()
        sys.exit(1)

    if args[0] == "learn":
        learn(opts)