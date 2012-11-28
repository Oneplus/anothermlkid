class MaxentData(object):
    """
    """
    # read in instances from file
    # initialize the featureDict and labelDict at the same time
    def __init__(self, filename):
        try:
            fp=open(filename, "r")
        except:
            raise IOError("failed to open file")

        self.instances = []
        self.labelDict = {}
        self.featDict  = {}
        self.param = {}
        self.N = 0

        for line in fp:
            sep = line.strip().split()
            label = sep[0]
            feats = sep[1:]

            # insert label into label dict
            if label not in self.labelDict:
                self.labelDict[label] = len(self.labelDict)+1

            # insert feat into feat dict
            for feat in feats:
                if feat not in self.featDict:
                    self.featDict[feat] = len(self.featDict)+1
                if feat not in self.param:
                    self.param[feat] = {}
                self.param[feat][label] = 0.0

            # append instance (x,y) into instances
            self.instances.append((label, feats))
            self.N += len(feats)

    def debug(self):
        print "instance:"
        for inst in self.instances:
            print inst.__str__()
        print "label dict:"
        for label in self.labelDict:
            print label.__str__()
        print "feat dict:"
        for feat in self.featDict:
            print feat.__str__()
        print "param:"
        for label in self.param:
            print "%s: %s" % (label, self.param[label].__str__())

    def f(self, label, feat):
        if feat in self.param and label in self.param[feat]: return 0
        else: return 1

    def p(self, label, feat):
        pass
