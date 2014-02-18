#!/usr/bin/env python
import sys
from GISTrainer   import GISTrainer
from lbfgsTrainer import lbfgsTrainer

class MaxentModel(object):
    '''
    The maxent model
    '''
    # init the model with certain training method
    def __init__(self, opts):
        if opts.trainMethod == "GIS":
            self.trainer = GISTrainer(opts.iteration,
                                      opts.sigma2,
                                      opts.tol,
                                      opts.alpha)
        elif opts.trainMethod == "lbfgs":
            self.trainer = lbfgsTrainer()

    # invoke the trainer to conduct the training
    # process
    def train(self, data):
        self.trainer.train(data)


    def test(self, data):
        try:
            pass
        except:
            pass


    def load_model(self, filename):
        pass


    def save_model(self, filename):
        pass

if __name__=="__main__":
    print >> sys.stderr, "library is not runnable."
