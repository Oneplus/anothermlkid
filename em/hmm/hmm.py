#!/usr/bin/env python
'''
A Un-supervised HMM trainer and tagger
'''
import sys
import itertools
import logging

from collections import defaultdict
from numpy import array, zeros, ones, log, exp

FORMAT = "[%(levelname)5s] %(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M:%S")

_NINF = float('-inf')

def logsumexp(a):
    '''
    - param     a   array like
    '''
    max_element = a.max()
    return max_element + log(exp(a - max_element).sum())

def read_sequence(context):
    '''
    An example of the context

        cat N
        chase V
        dog N

    - return list(list) the sequence
    '''
    return [l.split()[:2] for l in context.strip().split("\n")]

def flatten(l2d):
    '''
    Flatten list of list into list

    - param     list(list)  l2d
    - return    list        the list
    '''
    return list(itertools.chain(*l2d))

def words(sequence):
    '''
    Extract words from sequence
    '''
    return [token[0] for token in sequence]

def tags(sequence):
    '''
    Extract tags from instance
    '''
    return [token[1] for token in sequence]

def compute(PRIORS, EMITS, TRANS, states, sequence):
    '''
    Previously cache
    - log p(xt|yt=j)
    - log p(yt=j|yt-1=k)
    '''
    L = len(sequence)
    T = len(states)

    g0 = ones(T, dtype=float) * _NINF
    g = ones((L, T, T), dtype=float) * _NINF

    token = sequence[0]
    for j, state in enumerate(states):
        g0[j] = log(PRIORS[state]) + log(TRANS[token, state])

    for i in range(1, L):
        token = sequence[i]
        for j, state in enumerate(states):
            for k, pstate in enumerate(states):
                g[i,k,j] = log(EMITS[state, token]) + log(TRANS[pstate, state])

    return g0, g

def forward(g0, g, states, sequence):
    '''
    '''
    L = len(instance)
    T = len(states)

    a = zeros((L, T), dtype=float)
    a[0,:] = g0

    for i in xrange(1, L):
        for j in xrange(T):
            a[i,j] = logsumexp(a[i-1,:] + g[i,:,j])

    return a

def backward(g, states, sequence):
    '''

    '''
    L = len(sequence)
    T = len(states)

    b = zeros((L, T), dtype=float)
    b[L-1,:] = 0.
    for i in reversed(xrange(L - 1)):
        for j in xrange(T):
            b[i,j] = logsumexp(b[i+1,:] + g[i,j,:])

    return b

def argmax(g0, g, L, T):
    '''
    '''
    s = zeros((L, T), dtype=float)
    p = zeros((L, T), dtype=int)

    s[0] = g0
    p[0] = array([-1] * T)

    for i in range(1, L):
        for t in range(T):
            s[i,t] = (s[i-1,] + g[i,:,t]).max()
            p[i,t] = (s[i-1,] + g[i,:,t]).argmax()

    return s, p

def viterbi(g0, g, instance):
    '''
    '''
    L = len(instance)
    T = len(TAGS)

    s, p = argmax(g0, g, L, T)

    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret

def train_unsupervised(PRIORS, EMITS, TRANS,
                       symbols, states,
                       unlabeled_sequences,
                       max_iteration = 1000,
                       epsilon = 1e-6):
    '''
    The EM algorithm

    - PRIORS: the priors
    - EMITS : the emission probability
    - TRANS : the transition probability
    - symbols:  the collection of symbols
    - states:   the collection of states
    '''

    N = len(symbols)
    M = len(states)

    converge = False

    last_likelihood = None
    iteration = 0

    while not converge and iteration < max_iteration:
        counts_alpha = defaultdict(float)
        counts_eta = defaultdict(float)

        marginal_counts_alpha = defaultdict(float)
        marginal_counts_eta = defaultdict(float)

        #N = len(train_data)

        likelihood = 0.

        for sequence in unlabeled_sequences:
            sequence = words(sequence)

            L = len(sequence)

            g0, g = compute(PRIORS, EMITS, TRANS, states, sequence)

            a = forward(g0, g, states, sequence)
            b = backward(g, states, sequence)

            logZ = logsumexp(a[L-1,:])
            likelihood += logZ

            for i, word in enumerate(instance):
                if i == 0:
                    for j, tag in enumerate(TAGS):
                        marginal_probability = a[i,j] * b[i,j] / Z

                        counts_alpha[word, tag] += marginal_probability
                        marginal_counts_alpha[word, tag] += marginal_probability
                else:
                    for j, tag in enumerate(TAGS):
                        for k, ptag in enumerate(TAGS):
                            marginal_probability = a[i,j] * b[i,j] / Z
                            counts_alpha[word, tag] += marginal_probability
                            marginal_counts_alpha[tag] += marginal_probability

                            marginal_probability = a[i-1,k] * g[i-1,k,j] * b[i,j] / Z
                            counts_eta[ptag, tag] += marginal_probability
                            marginal_counts_eta[ptag] += marginal_probability

        for word, tag in EMITS.keys():
            EMITS[word, tag] = counts_alpha[word, tag] / marginal_counts_alpha[tag]

        for ptag, tag in TRANS.keys():
            TRANS[ptag, tag] = counts_eta[ptag, tag] / marginal_counts_eta[ptag]

        if last_likelihood and abs(last_likelihood - likelihood) < epsilon:
            converge = True

        print iteration, likelihood
        iteration += 1

    return PRIORS, EMITS, TRANS

def train_supervised(labeled_sequences, symbols, states):
    '''
    Use labeled data to train a model. Some smooth method should be applied.

    - return: PRIORS
              TRANS
              EMITS
    '''
    PRIORS = defaultdict(float)
    TRANS = defaultdict(float)
    EMITS  = defaultdict(float)

    # The initialize process
    counts = defaultdict(float)

    for sequence in labeled_sequences:
        pstate = None
        for i, (symbol, state) in enumerate(sequence):
            counts[symbol, state] += 1
            counts[pstate, state] += 1
            counts[state] += 1
            pstate = state

    # init the parameters
    for symbol in symbols:
        for state in states:
            if (symbol, state) in counts:
                EMITS[symbol, state] = (counts[symbol, state] + 1)/ counts[state]
            else:
                EMITS[symbol, state] = 0.

    for state in states:
        PRIORS[state] = counts[None, state] / len(labeled_sequences)

    for pstate, state in itertools.product(states, states):
        TRANS[pstate, state] = counts[pstate, state] / counts[pstate]

    return PRIORS, EMITS, TRANS


def parse_cmdline():
    '''
    Parse the command line options, which is used to specify supervised/unsupervised
    training data, testing data.

    - return    opts, args  result given by the OptionParser
    '''
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option("-s", dest="labeled",
                      help="use to specify the supervised data")
    parser.add_option("-d", dest="unlabeled",
                      help="use to specify the unsupervised data")
    parser.add_option("-t", dest="test",
                      help="use to specify the test data")

    return parser.parse_args()

if __name__=="__main__":
    opts, args = parse_cmdline()

    if opts.labeled is None:
        # If the tagged data is not provided, init with uniform distribution
        logging.warn("Tagged data is not specify")
        logging.warn("Init parameters with uniform distribution")

        labeled_sequences = []
    else:
        try:
            labeled_handle = open(opts.labeled, "r")
        except IOError:
            logging.warn("Failed to open tagged file %s." % opts.labeled)
            sys.exit(1)

        labeled_sequences = [read_sequence(i) \
                for i in labeled_handle.read().strip().split("\n\n")]

    try:
        unlabeled_handle = open(opts.unlabeled, "r")
    except:
        logging.warn("Failed to open untagged file %s." % opts.unlabeled)
        sys.exit(1)

    unlabeled_sequences = [read_sequence(i) \
            for i in unlabeled_handle.read().strip().split("\n\n")]

    # construct the lexicon
    symbols = set(flatten([words(instance) \
            for instance in labeled_sequences + unlabeled_sequences]))
    states = ['O', 'I-GENE']

    logging.info("Number of symbols is %d" % len(symbols))
    logging.info("Number of states is %d" % len(states))

    PRIORS, EMITS, TRANS = \
            train_supervised(labeled_sequences, symbols, states)
    PRIORS, EMITS, TRANS = \
            train_unsupervised(PRIORS, EMITS, TRANS, \
            symbols, states, \
            unlabeled_sequences)

    try:
        fp=open(sys.argv[2], "r")
    except:
        logging.error("Failed to open file")
        sys.exit(1)

    test_data = [read_instance(i) for i in fp.read().strip().split("\n\n")]

    for instance in test_data:
        g0, g = compute(EMITS, TRANS, instance)

        print viterbi(g0, g, instance)
