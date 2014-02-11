Unsupervised HMM
---

HMM model can be solved by EM algorithm unsupervisedly.

Since HMM model the probability of $p(y, x) = p(y) p(x | y)$

the likelihood of a given instance x can be write as:

![likelihood](http://latex.codecogs.com/gif.latex?p(x)=\sum_y{p(y, x)})

### Math Work

### Forward-backward

Model the probability $p(x\_1, ..., x\_{j-1}, y\_j = q)$

![forward](http://latex.codecogs.com/gif.latex?p(x_1,...,x_{j-1},y_j=q)=\sum_p{p(x_1,...,x_{j-1},y\_{j-1}= p,y_j=q)})

### Use logarithm

Using logarithm to replace the chained multiple can reduce errors.

### Experiments

The bio term NER data in NLP class is choosen as data set in this component.

`nltk.tag.hmm` is used as some kind of benchmark.
It is also used as a validation of the correctness for my implementation.

In the experiments, I found that if we apply the fully unsupervised learning,
the algorithm can't obtain a good result on `gene.dev` data. An alternative
solution is initialize the parameters with a small set of supervised data.

There are 13,795 instances in `./gene.train`. I take 1,000 instances to
perform the initial supervised learning. Rest is treated as unsupervised
data.

