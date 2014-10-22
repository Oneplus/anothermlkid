#!/usr/bin/env python

class Ngram(object):
    def __init__(self, *args):
        self.token = "||".join(args)

    def tokens(self):
        return self.tokens.split("||")

    def __hash__(self):
        return self.token.__hash__()

    def __str__(self):
        return ", ".join(self.token.split("||"))

    def __repr__(self):
        return str(self)
