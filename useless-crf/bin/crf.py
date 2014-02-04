#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(ROOTDIR)

from useless.logger import INFO, WARN, ERROR, LOG, LOG2
from useless.model import crfl2sgdmodel
from useless.l2sgd import l2sgd
from useless.lbfgs import lbfgs
from useless.viterbi import viterbi
from useless.instance import Instance

def evaluate(model, eval_file):
    fp = open(eval_file, "r")
    instances = [Instance(c) for c in fp.read().strip().split("\n\n")]

    nr_correct = 0
    nr_tags = 0
    for instance in instances:
        predict = viterbi(model, instance)
        for index, word in enumerate(instance.raw):
            if predict[index] == model.tags.get(word[0], 0):
                nr_correct += 1

        nr_tags += len(instance)

    acc = float(nr_correct) / nr_tags
    LOG2(INFO, "accuracy = %f (%d/%d)" % (acc, nr_correct, nr_tags))


def learn(opts):
    try:
        fp = open(opts.train_file, "r")
    except IOError:
        LOG(ERROR, "Failed to open train file %s:" % opts.train_file)
        return
    except:
        return

    m = crfl2sgdmodel()
    instances = [Instance(c) for c in fp.read().strip().split("\n\n")]
    fp.close()

    m.preprocess(instances)

    LOG(INFO, "number of tags %d" % m.nr_tags)
    LOG(INFO, "number of attributes %d" % m.nr_attrs)
    LOG(INFO, "number of instances %d" % len(instances))
    LOG(INFO, "paramter dimision is %d" % m.nr_dim)

    if opts.algorithm == "l2sgd":
        for i in xrange(opts.epoth):
            l2sgd(model              = m,
                  instances          = instances,
                  nr_epoth           = opts.cepoth,
                  init_learning_rate = opts.eta)

            if opts.dev_file:
                evaluate(m, opts.dev_file)

    elif opts.algorithm == "lbfgs":
        lbfgs(model     = m,
              instances = instances)

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
    optparser.add_option("-v", "--verbose", dest="verbose",
                         action="store_true", default=False,
                         help="verbose output")

    opts, args = optparser.parse_args()

    if opts.verbose:
        import useless.logger
        useless.logger.__VERBOSE__ = True

    if len(args) == 0 or args[0] not in ["learn", "tag"]:
        print >> sys.stderr, "ERROR: command [learn/tag] must be specified.\n"
        optparser.print_help()
        sys.exit(1)

    if args[0] == "learn":
        learn(opts)
