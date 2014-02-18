#!/usr/bin/env python
import sys
import logging

__VERBOSE__ = False

INFO  = 0
WARN  = 1
ERROR = 2

FORMAT = "[%(levelname)5s] %(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT,
                    level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M:%S")

def LOG(lvl, msg):
    if __VERBOSE__:
        if lvl == INFO:
            logging.info(msg)
        elif lvl == WARN:
            logging.warn(msg)
        elif lvl == ERROR:
            logging.error(msg)


def LOG2(lvl, msg):
    if lvl == INFO:
        logging.info(msg)
    elif lvl == WARN:
        logging.warn(msg)
    elif lvl == ERROR:
        logging.error(msg)


def trace(msg, index, N):
    if not __VERBOSE__:
        return

    percentage = (index + 1) * 100 / N

    pp = index * 50 / N
    bar = "%-20s |" % msg
    bar += ("#" * pp) + (" " * (50 - pp)) + "|" + ("%3d%%" % percentage)
    if index == N - 1:
        sys.stderr.write(bar + "\r\n")
    else:
        sys.stderr.write(bar + "\r")
