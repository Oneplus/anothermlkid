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
