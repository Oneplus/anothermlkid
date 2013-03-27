#!/usr/bin/env python
from optparse import OptionParser, make_option

if __name__=="__main__":

    option_list = [
            make_option("--debug",  dest="debug_mode", action="store_true", default=True),
            make_option("--train",  dest="train_mode", action="store_true", default=False),
            make_option("--one-iter", dest="one_iter", action="store_true", default=False),
            make_option("--valid",  dest="valid_file"),
            make_option("--nbest",  dest="nbest", action="store_true", default=False),
            make_option("--test",   dest="test_file"),
            make_option("--class",  dest="class", type="int"),
            make_option("--lambada", dest="lambda", type="float", default=0.75),
            make_option("--gradient-cutoff", dest="gradient_cutoff", type="float", default=15.),
            make_option("--dynamic", dest="dynamic", type="float", default=0.),
            make_option("--gen", dest="gen", type="int", default=0),
            make_option("--independent", dest="independent", action="store_true", default="False"),
            make_option("--alpha", dest="starting_alpha", type="float", default=0.1),
            make_option("--beta", dest="regularization", type="float", default=1e-7),
            #make_option("--min-improvement"),
            #make_option("--anti-kasparek"),
            make_option("--hidden", dest="hidden_size", type="int", default=30),
            make_option("--compress", dest="compress_size", type="int", default=0),
            make_option("--direction", dest="direct", type="int", default=0),
            make_option("--direct-order", dest="direct_orders", type="int", default=3),
            make_option("--bptt", dest="bptt", type="int", default=0),
            make_option("--bptt-block", dest="bptt_block", type="int", default=10),
            make_option("--random-size", dest="rand_seed", type="int", default=1),
            make_option("--lm-prob", dest="lmprob_file"),
            make_option("--binary", dest="fileformat", action="store_true", default=False),
            make_option("--rnnlm", dest="rnnlm_file"),]

    opt_parser = OptionParser(option_list=option_list)
    opts, args = opt_parser.parse_args()

    if opts.train_mode:
        model = RnnModel(opts)
        model.trainNet()
