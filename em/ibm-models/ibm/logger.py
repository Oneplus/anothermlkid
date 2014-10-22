#!/usr/bin/env python
import sys
import logging

INFO  = logging.INFO
WARN  = logging.WARNING
ERROR = logging.ERROR

FORMAT = "[%(levelname)5s] %(asctime)-15s %(message)s"
logging.basicConfig(format = FORMAT,
                    level = logging.INFO,
                    datefmt = "%Y-%m-%d %H:%M:%S")

def LOG(lvl, msg):
    logging.log(lvl, msg)


def TRACE(msg, index, N):
    percentage = (index + 1) * 100 / N
    pp = index * 50 / N
    bar = "%-20s |" % msg
    bar += ("#" * pp) + (" " * (50 - pp)) + "|" + ("%3d%%" % percentage)
    if index == N - 1:
        sys.stderr.write(bar + "\r\n")
    else:
        sys.stderr.write(bar + "\r")
