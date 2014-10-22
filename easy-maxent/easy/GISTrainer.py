#!/usr/bin/env python
import sys
import math

def msg(s):
    print >> sys.stderr, s

class GISTrainer(object):
    def __init__(self, _maxIter, _sigma2, _tol, _alpha):
        self.maxIter = _maxIter
        self.sigma2 = float(_sigma2)
        self.eps = float(_tol)
        self.alpha = float(_alpha)

    def train(self, data):
        # prepare
        self.n = float(len(data.instances))
        # calculate slow factor
        self.C = float(self._cal_C(data))

        # calculate E_{\title{p}}f_i
        observation = self._cal_EEp(data)

        # for debug
        # for ob in observation:
        #     print ob, observation[ob]

        prelikelihood = -1e10

        msg("")
        msg("Starting GIS iterations ...")
        msg("Number of Predicates: %d" % len(data.param))
        msg("Number of Outcomes:   %d" % len(data.labelDict))
        msg("Number of Parameters: %d" % 0)
        msg("Tolerance:            %f" % self.eps)
        msg("Gassian Penalty:      %s" % ("on" if self.sigma2 > 0.0 else "off"))
        msg("")
        msg("iters   loglikelihood   training accuracy   heldout accuracy")
        msg("============================================================")

        for it in xrange(self.maxIter):
            likelihood, expection = self._cal_Ep(data)

            for f in data.param:
                for c in data.param[f]:
                    inc = 0.0

                    # apply Gaussian prior smoothing
                    if self.sigma2 > 0.0:
                        inc = self._newton(expection[f][c],
                                           observation[f][c],
                                           data.param[f][c],
                                           self.eps)
                    # GIS ?
                    elif self.alpha > 0.0:
                        if observation[f][c] - self.alpha <= 0.0:
                            continue

                        inc = (math.log(observation[f][c]-self.alpha)
                                -math.log(observation[f][c]))/self.C
                        if data.param[f][c] + inc <= 0.0:
                            inc = -data.param[f][c]
                    # Standard GIS
                    else:
                        inc = (math.log(observation[f][c])
                                -math.log(observation[f][c]))/self.C

                    data.param[f][c] += inc

            msg("%-7d %-16f" % (it+1, likelihood))
            #for f in data.param:
            #    print f, data.param[f]

            if abs((prelikelihood - likelihood) / prelikelihood) < self.eps:
                break
            prelikelihood = likelihood


    # calculate the C value/ slow factor of certain data
    # in GIS the iteration process in show below:
    #   \lambda_i^{(t+1)}=\lambda_i^{(t)}+\frac{1}{C}log{\frac{E_{\tilde{p}}f_i}{E_{p^{(t)}}f_i}}
    # where
    #   C=\max_{x,y}\sum_{i=1}^nf_i(x,y)
    def _cal_C(self, data):
        maxC = 0
        for inst in data.instances:
            C = 0
            label, feats = inst

            for feat in feats:
                if label in data.param and feat in data.param[label]:
                    C += 1
            maxC = max(maxC, C)
        return maxC


    # calculate the model expection
    #   E_{p}f_{i}=\sum_{x,y}\widetilde{p}(x)p(y|x)f_{i}(x,y)
    # where
    #   \widetitle{p}(x)=1/\sum_{x,y}{C(x,y)}
    # p(y|x) can be calculate from the maximum entropy formula
    def _cal_Ep(self, data):
        expection={}
        loglikelihood=0.0
        for event in data.instances:
            label, features = event
            prob={}
            for c in data.labelDict: prob[c] = 0.0

            for f in features:
                for c in data.param[f]:
                    try:
                        prob[c] += data.param[f][c]
                    except KeyError:
                        prob[c] = data.param[f][c]

            # max_prob = max([prob[c] for c in data.labelDict])
            for c in prob: prob[c]=math.exp(prob[c])
            tot=sum([prob[c] for c in data.labelDict])
            for c in data.labelDict:
                prob[c]=prob[c]/tot
            # print prob
            for f in features:
                if f not in expection:
                    expection[f]={}
                for c in data.param[f]:
                    try:
                        expection[f][c] += prob[c]
                    except KeyError:
                        expection[f][c] = prob[c]

            loglikelihood+=math.log(prob[label])
        return (loglikelihood, expection)


    # calculate empirical expection
    #   E_{\widetitle{p}}f_{i}=\sum_{x,y}\widetitle{p}(x,y)f_i(x,y)
    # where
    #   \widetitle{p}(x,y)=C(x,y)/\sum_{x,y}{C(x,y)}
    # if we don't count \sum_{x,y}{C(x,y)} in empirical expection
    # we don't need take \sum_{x,y}{C(x,y)} in model expection either
    def _cal_EEp(self, data):
        observation={}
        for event in data.instances:
            label, features = event
            for f in features:
                if f not in observation:
                    observation[f]={}
                if label not in observation[f]:
                    observation[f][label]=0.0
                observation[f][label]+=1.0
        return observation


    #
    #   $ E_{\title{p}}f_i=E_pf_ie^{C\delta_i}+frac{\lambda_i+\delta_i}{\sigma_i^2}
    def _newton(self, f_q, f_ref, lambda_i, eps):
        # print "in newton f_q=%f f_ref=%f" % (f_q, f_ref)
        maxIter=50
        x0=0.0
        x=0.0

        for iter in xrange(maxIter):
            t=f_q*math.exp(self.C * x0)
            fval = t + self.n * (lambda_i + x0)/self.sigma2 - f_ref
            fpval = t * self.C + self.n / self.sigma2

            if fpval == 0:
                msg("WARNING: zero-division encounter in newtown() method")
                return x0

            x = x0 - fval / fpval
            if abs(x - x0) < eps:
                return x
            x0 = x

        msg("ERROR: newtown method failed.")
        raise NumberError("newtown method failed.")


if __name__=="__main__":
    msg("library is not runnable.")
