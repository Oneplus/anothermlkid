#!/usr/bin/env python
import sys
import os
import itertools
import gzip
import cPickle as pkl

from collections import defaultdict

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
if ROOTDIR not in sys.path:
    sys.path.append(ROOTDIR)

from ibm.ngram  import Ngram
from ibm.logger import INFO, WARN, ERROR, LOG, TRACE

class Model1(object):
    def __init__(self, file_e, file_f, n, max_iter):
        '''
        Initialize with parallel corpus
        '''
        try:
            handle_e = open(file_e, "r")
            handle_f = open(file_f, "r")
        except IOError:
            print >> sys.stderr, "Failed to open file."
            sys.exit(1)
        except:
            sys.exit(1)

        self.max_iter = max_iter

        def read_corpus(fp):
            return [l.strip().split() for l in fp]

        def flatten(l2d):
            return list(itertools.chain(*l2d))

        self.corpus_e = ce = read_corpus(handle_e)[:n]
        self.corpus_f = cf = read_corpus(handle_f)[:n]

        LOG(INFO, "Read in %d english instances" % len(ce))
        LOG(INFO, "Read in %d franch instances" % len(cf))

        self.dict_e = set(flatten(ce))
        self.dict_f = set(flatten(cf))

        LOG(INFO, "Size of English vocabulary %d" % len(self.dict_e))
        LOG(INFO, "Size of French vocabulary %d" % len(self.dict_f))

        self.trans = defaultdict(float)
        self.align = defaultdict(float)


    def em(self):
        # initialize
        M = len(self.dict_e)
        t = self.trans
        LOG(INFO, "Initializing translation matrix")
        for index, ej in enumerate(self.dict_e):
            TRACE("Initializing", index, M)
            for fk in self.dict_f:
                ngram = Ngram(ej, fk)
                t[ngram] = 1. / M

        LOG(INFO, "Initialize t is done")

        for i in xrange(self.max_iter):
            self._em(i)
            LOG(INFO, "EM for iteration %d is done." % (i+1))


    def _em(self, iteration=0):
        trans_counts = defaultdict(float)
        words_counts = defaultdict(float)

        align_counts = defaultdict(float)
        posit_counts = defaultdict(float)

        t = self.trans
        q = self.align

        Ne = len(self.corpus_e)
        # E-step
        LOG(INFO, "E-step of iteration %d" % (iteration+1))
        for index, (e, f) in enumerate(zip(self.corpus_e, self.corpus_f)):
            l, m = len(e), len(f)

            TRACE("E-step", index, Ne)
            for i, fi in enumerate(f):
                sum_t = sum([t[Ngram(ej, fi)] for ej in e])

                for j, ej in enumerate(e):
                    ngram = Ngram(ej, fi)
                    delta = t[ngram] / sum_t
                    trans_counts[ngram] += delta
                    words_counts[ej] += delta
                    align_counts[j, i, l, m] += delta
                    posit_counts[i, l, m] += delta

        # M-step
        LOG(INFO, "M-step of iteration %d" % (iteration+1))
        for k, v in trans_counts.iteritems():
            ej, fi = k.split()
            t[k] = trans_counts[k] / words_counts[ej]

        for k, v in align_counts.iteritems():
            j, i, l, m = k
            q[k] = align_counts[k] / posit_counts[i, l, m]



    def save(self, filename):
        pkl.dump((self.trans, self.align),
                 gzip.open(filename, "wb"), pkl.HIGHEST_PROTOCOL)


if __name__=="__main__":
    print >> sys.stderr, "Library is not runnable"
