#!/usr/bin/env python
import nltk

def read(filename):
    train_text = open(filename).read().strip()
    sequences = [[token.split()[:2] for token in inst.split("\n")] \
            for inst in train_text.split("\n\n")]

    return sequences

sequences = read("./gene.test")

symbols = list(set([token[0] for sequence in sequences for token in sequence]))
states = ['O', 'I-GENE']

trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)

print "Start to train."

m = trainer.train_unsupervised(sequences)

print "Train is done."

sequences = read("./gene.key")

print "Start to test."

nr_correct = 0
nr_label = 0

for sequence in sequences:
    tags = m.tag([token[0] for token in sequence])
    print tags
    print [tkn[1] for tkn in sequence]
    nr_correct += len([1 for pt, ct in zip(tags, sequence) if pt[1] == ct[1]])
    nr_label += len(tags)

print "%f , (%d / %d)" % (float(nr_correct) / nr_label, nr_correct, nr_label)

print "Test is done."
