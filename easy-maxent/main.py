#!/usr/bin/env python
import sys
from optparse    import OptionParser
from maxentModel import MaxentModel
from maxentData  import MaxentData

def main():
    usage = ""
    optParser = OptionParser(usage)
    optParser.add_option("-f", "--file", dest="dataFile")
    optParser.add_option("-m", "--model", dest="modelFile")
    optParser.add_option("-a", "--method", dest="trainMethod", default="GIS")
    optParser.add_option("-i", "--iter", dest="iteration", default=30)
    optParser.add_option("-t", "--tol", dest="tol", default=1e-3)
    optParser.add_option("-s", "--sigma2", dest="sigma2", default=100)
    optParser.add_option("-p", "--alpha", dest="alpha", default=0)

    opts, args = optParser.parse_args()

    # handle error parameter
    if len(args) < 0:
        print >> sys.stderr, "ERROR: command[learn/tag] must be specified."
        return

    # the learning process
    if args[0] == "learn":
        try:
            data = MaxentData(opts.dataFile)
        except:
            print >> sys.stderr, "failed to load data file"

        #data.debug()

        model = MaxentModel(opts)
        model.train(data)
        #model.save_model(opts.modelFile)

    # the tagging process
    elif args[0] == "tag":
        try:
            data = MaxentData(opts.dataFile)
        except:
            print >> sys.stderr, "failed to load data file"

        model = MaxentModel(opts.trainMethod)
        #model.load_model(opts.modelFile)
        #model.test(data)
    else:
        print >> sys.stderr, "unknown mode"

if __name__=="__main__":
    main()
