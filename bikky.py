#!/usr/bin/env python3
"""A bare-bones program to help you identify broken biscuits, to be used in
conjunction with CE316/CE866 experiments.  The processing is as follows:

  1. the image is read in as grey-scale
  2. it is thresholded using a fixed value
  3. the resulting binary image is tidied up using morphological operations
  4. contours are found around each foreground object
  5. each contour is processed
  6. some text is written on the image

You have two jobs to do:
  + Improve the thresholding stage so it identifies individual
    biscuits more reliably.

  + Determine whether a biscuit is circular, rectangular or broken.
"""

import sys, argparse, datetime, time, numpy, cv2, sxcv, taskfile


def process_biscuit (fn, gt, display, threshold=100, mask_size=9):
    """Process a single image file, returning the outcome.  If verbose is
    True, output information and display the result."""

    # Read in the image and binarize it.
    im = cv2.imread (fn, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print ("Cannot read " + fn, file=sys.stderr)
        exit (2)
    bim = sxcv.binarize (im, threshold, 0, 255)

    # Tidy up the binary image by deleting small regions and filling in gaps.
    kernel = numpy.ones ((mask_size, mask_size), numpy.uint8)
    bim = cv2.morphologyEx (bim, cv2.MORPH_OPEN, kernel)
    bim = cv2.morphologyEx (bim, cv2.MORPH_CLOSE, kernel)

    # Find contours and print them out.
    contours, _ = cv2.findContours (bim, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Calculate features that will categorize a biscuit's contour as
    # "circ" (circular), "rect" (rectangular) or "rej" (broken) and
    # store the result in the variable "outcome".
    outcome = "rej"

    # Print the outcome in the format required by FACT.
    print ("result %s %s S %s" % (fn, gt, outcome))

    if display:
        # Write the outcome on the image.
        cv2.putText (im, outcome, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                     (255,255,255), 2, cv2.LINE_AA)
        # Draw the contours on the image and display the result.
        cv2.drawContours (im, contours, -1, 0, 2)
        cv2.imshow (fn, im)
        k = cv2.waitKey (0)
        if k == ord ("q") or k == ord ("Q"):
            exit (0)


#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Set up the parsing of the command line.
clp = argparse.ArgumentParser (description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
clp.add_argument ("-d", action="store_true", default=False,
                  help="display each image as it is processed")
clp.add_argument ("-dataset", default="train",
                  help="dataset to use when processing a taskfile")
clp.add_argument ("files", nargs="+", default="",
                  help="the files to be processed")

# Parse the command line and do a little post-processing to make our life
# easier below.
args = clp.parse_args()
if isinstance (args.files, str):
    args.files = [args.files]

# If we were invoked with a single command-line argument and it ends in ".task",
# we assume it is an ELVS-compatible task file and load that.   Otherwise, it
# should the image files to be used.
if len (args.files) == 1 and args.files[0].endswith (".task"):
    task = taskfile.load (args.files[0])

    # We shall output our results in a form which can be processed by FACT.
    # The first output line contains a timestamp.
    now = datetime.datetime.now().isoformat()
    print ("transcript_begin %s %s %s %s" % \
           (task["name"], "0.00", task["type"], now[:-7]))

    # We want to include in the trailer how long the tests took to do, so we
    # need to start and stop a timer outside the per-image loop.
    start_time = time.perf_counter ()
    for f, c in task[args.dataset]:
        process_biscuit (f, c, args.d)
    tix = time.perf_counter() - start_time
    print ("transcript_end %.2f" % tix)

else: 
    # It's a list of filenames, so we just process them.
    for fn in args.files:
        process_biscuit (fn, "?", args.d)
    cv2.destroyAllWindows ()
