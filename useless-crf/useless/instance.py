from collections import defaultdict

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

        self.features_table = None
        self.correct_features = None

    def __len__(self):
        return len(self.raw)

def build_instance(w, _attrs, _tags, instance, train=True):
    '''
    Build instance from raw instance

    - param[in] w       the parameters vector
    - param[in] attrs   dict of attributes
    - param[in] tags    dict of tags
    '''
    instance.features_table = f = {}
    instance.correct_features = F = defaultdict(int)

    T = len(_tags)
    A = len(_attrs)

    previous_idx = None
    for i, item in enumerate(instance.raw):
        tag, attrs = item
        attrs = [_attrs[attr] for attr in attrs if attr in _attrs]
        if i == 0:
            for k in xrange(T):
                f[i,None,k] = [attr * T + k for attr in attrs]
        else:
            for j in xrange(T):
                for k in xrange(T):
                    f[i,j,k] = [attr * T + k for attr in attrs]
                    f[i,j,k].append((A + j) * T + k)

        if train:
            idx = _tags[tag]
            for attr in attrs:
                F[attr * T + idx] += 1
            if i > 0:
                F[(A + previous_idx) * T + idx] += 1
            previous_idx = idx
