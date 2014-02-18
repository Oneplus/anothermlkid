from itertools import chain
from collections import defaultdict


try:
    from numpy import array
except ImportError:
    print >> sys.stderr, "scipy is not installed"
    sys.exit(1)


class Instance(object):
    def __init__(self, context):
        '''
        Content example

          \n
          tag1 attr1,1 attr1,2 ...
          tag2 attr2,1 attr2,2 ...
          \n
        '''

        self.raw = []

        for line in context.split("\n"):
            tokens = line.split()
            tag = tokens[0]
            attrs = tokens[1:]

            self.raw.append((tag, attrs))

        self.unigram_features_table = None
        self.bigram_features_table = None
        self.correct_features = None

    def __len__(self):
        return len(self.raw)

def build_instance(_attrs, _tags, instance, train=True):
    '''
    Build instance from raw instance

    - param[in]     attrs       dict of attributes
    - param[in]     tags        dict of tags
    - param[in/out] instance    the instance
    '''
    if (instance.unigram_features_table is not None and
            instance.bigram_features_table is not None):
        return

    instance.unigram_features_table = U = {}
    instance.bigram_features_table = B = {}
    instance.correct_features = F = defaultdict(int)

    T = len(_tags)
    A = len(_attrs)

    for i, item in enumerate(instance.raw):
        tag, attrs = item
        attrs = [_attrs[attr] for attr in attrs if attr in _attrs]
        for k in xrange(T):
            U[i,k] = array([attr * T + k for attr in attrs])

    for j in xrange(T):
        for k in xrange(T):
            B[j,k] = array([(A + j) * T + k])

    if train:
        j, k = None, _tags[instance.raw[0][0]]
        for e in U[0,k]:
            F[e] += 1
        for i, item in enumerate(instance.raw[1:]):
            j, k = k, _tags[item[0]]
            for e in chain(U[i+1,k].tolist(), B[j,k].tolist()):
                F[e] += 1

def destroy_instance(instance):
    instance.unigram_features_table = None
    instance.bigram_features_table = None
    instance.correct_features = None

