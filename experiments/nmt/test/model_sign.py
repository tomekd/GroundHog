#!/usr/bin/env python

import sys
import numpy
m = numpy.load(sys.argv[1])
for k, v in sorted(m.items()):
    print v.sum()
