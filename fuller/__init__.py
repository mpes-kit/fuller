#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils, metrics
import warnings as wn

with wn.catch_warnings():
    wn.simplefilter("ignore")
    wn.warn("deprecated", DeprecationWarning)
    wn.warn("future", FutureWarning)

try:
    from . import generator
except:
    pass

try:
    from . import mrfRec
except:
    pass


__version__ = '0.8.0'