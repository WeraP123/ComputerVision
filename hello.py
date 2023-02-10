#!/usr/bin/env python
"""hello.py -- a "hello world" program for sxcv and cv2"""
import sys, sxcv, cv2

# If a filename was given on the command line, read it in.  Otherwise, use
# the test image built into the sxcv module.
if len (sys.argv) > 1:
    im = cv2.imread (sys.argv[1])
    if im is None:
        print ("Couldn't read the image file '%s'!" % sys.argv[1],
               file=sys.stderr)
        exit (1)
    name = sys.argv[1]
else:
    im = sxcv.testimage ()
    name = "testimage"

# Output a one-line summary of the image.
print (sxcv.describe (im, name))
