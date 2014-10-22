#!/usr/bin/env python
import gzip
import cPickle as pkl
from logger import INFO, WARN, LOG

class Model2(object):
    def __init__(self, file_e, file_f):
        try:
            handle_e = open(file_e, "r")
            handle_f = open(file_f, "r")
        except IOError:
            print >> sys.stderr, "Failed to open file."
            sys.exit(1)
        except:
            sys.exit(1)

        def read_corpus(fp):
            return [l.strip().split() for l in fp]

        def flatten(l2d):
            return list(itertools.chain(*l2d))

        self.corpus_e = ce = read_corpus(handle_e)
        self.corpus_f = cf = read_corpus(handle_f)

        self.dict_e = set(flatten(ce))
        self.dict_f = set(flatten(cf))

    def em(self, initialize = None):
        if initialize is not None:
            self.trans, self.align = pkl.load(gzip.open(initialize, "rb"))
            LOG(INFO, "t and q is loaded.")
        else:
            self.trans = defaultdict(float)
            self.align = defaultdict(float)

            M = len(self.dict_e)
            t = self.trans
            for ej in self.dict_e:
                for fk in self.dict_f:
                    t[ej, fk] = 1. / M
            q = self.align
            for e, f in zip(self.corpus_e, self.corpus_f):
                l, m = len(e), len(f)
                for i, j in zip(xrange(l), xrange(m)):
                    q[j,i,l,m] = 1. / l

        for i in xrange(5):
            self._em()

    def _em(self):
        c1 = defaultdict(float)
        c2 = defaultdict(float)

        t = self.trans
        q = self.align

        # E-step
        for e, f in zip(self.corpus_e, self.corpus_f):
            l, m = len(e), len(f)

            for i, fi in enumerate(f):
                sum_t = sum([q[j, i, l, m] * t[ej, fi] for j, ej in enumerate(e)])

                for j, ej in enumerate(e):
                    delta = q[j, i, l, m] * t[ej, fi] / sum_t
                    c1[ej, fi]      += delta
                    c1[ej]          += delta
                    c2[j, i, l, m]  += delta
                    c2[i, l, m]     += delta

        # M-step
        for k, v in c1.iteritems():
            if isinstance(k, tuple):
                ej, fi = k
                t[k] = c1[k] / c1[ej]

        for k, v in c2.iteritems():
            if len(k) == 4:
                j, i, l, m = k
                q[k] = c2[k] / c2[i, l, m]

if __name__=="__main__":
    pass
