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

    - param[in]     w           the parameters vector
    - param[in]     attrs       dict of attributes
    - param[in]     tags        dict of tags
    - param[in/out] instance    the instance
    '''
    instance.features_table = f = {}
    instance.correct_features = F = defaultdict(int)

    T = len(_tags)
    A = len(_attrs)

    for i, item in enumerate(instance.raw):
        tag, attrs = item
        attrs = [_attrs[attr] for attr in attrs if attr in _attrs]
        if i == 0:
            for k in xrange(T):
                f[i,None,k] = [attr * T + k for attr in attrs]
        else:
            for k in xrange(T):
                base = [attr * T + k for attr in attrs]
                for j in xrange(T):
                    f[i,j,k] = base + [(A + j) * T + k]

    if train:
        j, k = None, _tags[instance.raw[0][0]]
        for e in f[0,None,k]: F[e] += 1
        for i, item in enumerate(instance.raw[1:]):
            j, k = k, _tags[item[0]]
            for e in f[i+1,j,k]:
                F[e] += 1

def destroy_instance(instance):
    instance.features_table = None
    instance.correct_features = None

