#!/usr/bin/env python
import sys
import os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
if ROOTDIR not in sys.path:
        sys.path.append(ROOTDIR)

from optparse    import OptionParser
from easy.maxentModel import MaxentModel
from easy.maxentData  import MaxentData

def main():
    usage = "./bin/maxent.py learn/tag [options]"
    optParser = OptionParser(usage)
    optParser.add_option("-f", "--file",
                         dest="dataFile",
                         help="use to specify the data file.")
    optParser.add_option("-m", "--model",
                         dest="modelFile",
                         help="use to specify the model file.")
    optParser.add_option("-a", "--method",
                         dest="trainMethod", default="GIS",
                         help="use to specify training method.")
    optParser.add_option("-i", "--iter",
                         dest="iteration", default=30,
                         help="use to specify maximum training iteration")
    optParser.add_option("-t", "--tol",
                         dest="tol", default=1e-3,
                         help="use to specify the iteration stop criterion")
    optParser.add_option("-s", "--sigma2",
                         dest="sigma2", default=100, type="float",
                         help="")
    optParser.add_option("-p", "--alpha",
                         dest="alpha", default=0, type="float",
                         help="")

    opts, args = optParser.parse_args()

    # handle error parameter
    if len(args) == 0 or args[0] not in ["learn", "tag"]:
        print >> sys.stderr, "ERROR (Parse Options): command[learn/tag] must be specified."
        optParser.print_help()
        return

    # the learning process
    if args[0] == "learn":
        try:
            data = MaxentData(opts.dataFile)
        except:
            print >> sys.stderr, "ERROR (Load Data): Failed to load data file"
            sys.exit(1)

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
