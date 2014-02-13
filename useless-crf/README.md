Useless CRF
-----------

A useless CRF toolkit.

* Since we have so many CRF toolkit, why you built it?
* I don't know ...

### Features

* Linear Chained CRF
    * can be used for chuncking, tagging and some other tasks
    * optimize with: lbfgs, l2sgd (under construction)

* Pairwised CRF (under construction)
    * can be used for word alignment in Machine Translation

* Gradient check

### Usage

Run the following commands to show a list of options.
```
./bin/crf.py --help
```

### Example

Example for training a model.

```
./bin/crf.py learn -t ./data/train.features -d ./data/test.features -a lbfgs
```

### Reference

* [An introduction to Conditional Random Fields for Relational Learning](http://arxiv.org/abs/1011.4088) Charpter 4 for Introduction to Statistical Relational Learning
* [Stochastic Gradient Descent Tricks](http://research.microsoft.com/pubs/192769/tricks-2012.pdf)

