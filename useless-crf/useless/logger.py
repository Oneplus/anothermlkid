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
            logging.warning(msg)
        elif lvl == ERROR:
            logging.error(msg)

def LOG2(lvl, msg):
    if lvl == INFO:
        logging.info(msg)
    elif lvl == WARN:
        logging.warning(msg)
    elif lvl == ERROR:
        logging.error(msg)

def trace(index, max_index = -1):
    if __VERBOSE__:
        if index + 1 == max_index:
            print >> sys.stderr, "%4d" % max_index
        elif (index + 1) % 100 == 0:
            print >> sys.stderr, "%4d" % (index + 1),
            if (index + 1) % 1000 == 0:
                print >> sys.stderr
