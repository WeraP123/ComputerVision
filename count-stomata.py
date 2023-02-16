#!/usr/bin/env python3
"""This operator-guided program counts the number of stomata in a leaf image,
stored in a file whose name is given on the command line.  The image is
displayed along with sliders that control the boundary between the colours of
leaf and stomata; normally, moving the "lo B" slider has the most effect.  As
the slider is moved, the stomata found are shown with black '+' marks and the
number of them shown on the image.  When you have found the stomata as well
as possible by moving the sliders, you can:

+ left-click at points on the image to add them to the count; they are marked
  with a red '+'

+ right-click (or shift-click) on black marks to remove them from the count;
  they are marked with a blue "x"

+ type "s" in the image window to save the marked image to a file in the
  current directory, the name of the input file prefixed with "counted-"

+ type " " (space) or "n" to move on to the next image

+ type "q" to quit

When you have finished with an image, its filename and stomata count are
written out, followed by the coordinates of the locations found or selected.
"""

#-------------------------------------------------------------------------------
# Boilerplate.
#-------------------------------------------------------------------------------

import sys, argparse, os
import cv2, sxcv

#-------------------------------------------------------------------------------
# Routines.
#-------------------------------------------------------------------------------

def draw_contour_centres (canvas, centres, colour, shape="+"):
    for cx, cy in centres:
        # Work out the limits of the "+" we want to draw.
        s = 5
        xlo = max (0, cx - s)
        xhi = cx + s
        ylo = max (0, cy - s)
        yhi = cy + s
        # Draw the marker.
        if shape == "+":
            cv2.line (canvas, (cx, ylo), (cx, yhi), colour)
            cv2.line (canvas, (xlo, cy), (xhi, cy), colour)
        else:
            cv2.line (canvas, (xlo, ylo), (xhi, yhi), colour)
            cv2.line (canvas, (xlo, yhi), (xhi, ylo), colour)


def find_contour_centre (c):
    "Return the centre of the contour."
    x, y, w, h = cv2.boundingRect (c)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def nearby (cx, cy, centres, threshold=10):
    # Look at all the other centres.
    dist_thresh = threshold**2
    for x, y in centres:
        if (x - cx)**2 + (y - cy)**2 < dist_thresh:
            return True

    # No centre is nearby.
    return False


def slider_event_handler (event, x, y, flags, param):
    global selected_centres, deleted_centres

    if event == cv2.EVENT_LBUTTONUP:
        if flags & cv2.EVENT_FLAG_CTRLKEY != 0:
            deleted_centres += [[x, y]]
        elif flags & cv2.EVENT_FLAG_SHIFTKEY != 0:
            deleted_centres += [[x, y]]
        else:
            selected_centres += [[x, y]]
    elif event == cv2.EVENT_RBUTTONUP:
        deleted_centres += [[x, y]]


def read_rgb_trackbar (junk):
    global lo_R, lo_G, lo_B, hi_R, hi_G, hi_B, me

    lo_R = cv2.getTrackbarPos ("lo R", me)
    lo_G = cv2.getTrackbarPos ("lo G", me)
    lo_B = cv2.getTrackbarPos ("lo B", me)
    hi_R = cv2.getTrackbarPos ("hi R", me)
    hi_G = cv2.getTrackbarPos ("hi G", me)
    hi_B = cv2.getTrackbarPos ("hi B", me)


def read_hsv_trackbar (junk):
    global lo_H, lo_S, lo_V, hi_H, hi_S, hi_V, me

    lo_H = cv2.getTrackbarPos ("lo H", me)
    lo_S = cv2.getTrackbarPos ("lo S", me)
    lo_V = cv2.getTrackbarPos ("lo V", me)
    hi_H = cv2.getTrackbarPos ("hi H", me)
    hi_S = cv2.getTrackbarPos ("hi S", me)
    hi_V = cv2.getTrackbarPos ("hi V", me)


#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Set up the parsing of the command line.
clp = argparse.ArgumentParser (description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
clp.add_argument ("-hsv", action="store_true", default=False,
                  help="work in HSV colour space rather than RGB")
clp.add_argument ("-minarea", type=int, default=10,
                  help="minimum region area in pixels")
clp.add_argument ("-maxarea", type=int, default=100,
                  help="maximum region area in pixels")

clp.add_argument ("files", nargs="+", default="",
                  help="the files to be processed")

# Parse the command line and do a little post-processing to make our life
# easier below.
args = clp.parse_args()
if isinstance (args.files, str):
    args.files = [args.files]

me = os.path.basename (sys.argv[0])
cv2.namedWindow (me)

# Give initial values to some variables that we'll use below.
if args.hsv:
    lo_H, lo_S, lo_V = 33, 0,0
    hi_H, hi_S, hi_V = 93, 100, 100
else:
    lo_R, lo_G, lo_B = 0, 0, 100
    hi_R, hi_G, hi_B = 255, 255, 255

# Having done all the preparation, we can now work through the images.
for fn in args.files:
    # Read in the image.
    im = cv2.imread (fn)
    if im is None:
        print ("Cannot read " + fn, file=sys.stderr)
        exit (1)
    if args.hsv:
        hsvim = cv2.cvtColor (im, cv2.COLOR_BGR2HSV)

    # Draw a set of sliders that allows the user to optimize the RGB or HSV
    # threshold values.  The user can also:
    #   right-click (or shift-click) to delete the nearest detected stomata
    #   left-click to add a location to the overall count
    if args.hsv:
        cv2.createTrackbar ("lo H", me, lo_H, 360, read_hsv_trackbar)
        cv2.createTrackbar ("hi H", me, hi_H, 360, read_hsv_trackbar)
        cv2.createTrackbar ("lo S", me, lo_S, 100, read_hsv_trackbar)
        cv2.createTrackbar ("hi S", me, hi_S, 100, read_hsv_trackbar)
        cv2.createTrackbar ("lo V", me, lo_V, 100, read_hsv_trackbar)
        cv2.createTrackbar ("hi V", me, hi_V, 100, read_hsv_trackbar)
    else:
        cv2.createTrackbar ("lo R", me, lo_R, 255, read_rgb_trackbar)
        cv2.createTrackbar ("hi R", me, hi_R, 255, read_rgb_trackbar)
        cv2.createTrackbar ("lo G", me, lo_G, 255, read_rgb_trackbar)
        cv2.createTrackbar ("hi G", me, hi_G, 255, read_rgb_trackbar)
        cv2.createTrackbar ("lo B", me, lo_B, 255, read_rgb_trackbar)
        cv2.createTrackbar ("hi B", me, hi_B, 255, read_rgb_trackbar)
    cv2.setMouseCallback (me, slider_event_handler)

    # Set up a few colours that we'll use when drawing.
    black = (  0, 0,   0)
    blue  = (255, 0,   0)
    red   = (  0, 0, 255)

    # List `selected_centres` will hold those stomata centres explicitly added
    # by the user, while list `deletec_centres` will hold those centres that
    # have been deleted by the user.
    selected_centres = []
    deleted_centres = []
    nf = 1

    # Loop until the user explicitly exits it.
    while True:
        # Make a copy of the image and mark selected and deleted centres.
        canvas = im.copy ()
        draw_contour_centres (canvas, selected_centres, red, "+")
        draw_contour_centres (canvas, deleted_centres, blue, "x")

        # Use the slider values to perform colour segmentation, then find
        # contours on the result.
        if args.hsv:
            lo = sxcv.hsv_to_cv2 (lo_H, lo_S, lo_V)
            hi = sxcv.hsv_to_cv2 (hi_H, hi_S, hi_V)
            bim = cv2.inRange (hsvim, lo, hi)
        else:
            lo = (lo_B, lo_G, lo_R)
            hi = (hi_B, hi_G, hi_R)
            bim = cv2.inRange (im, lo, hi)
        contours, _ = cv2.findContours (bim, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Make sure the contours we find are in the allowed size range and
        # are not too close to each other.
        valid_centres = []
        for c in contours:
            a = cv2.contourArea (c)
            if a >= args.minarea and a <= args.maxarea:
                cx, cy = find_contour_centre (c)
                centres = selected_centres + deleted_centres + valid_centres
                if nearby (cx, cy, selected_centres): continue
                if nearby (cx, cy, deleted_centres): continue
                if nearby (cx, cy, valid_centres): continue
                valid_centres += [[cx, cy]]

        # Draw the remaining contour centres onto a copy of the image.
        draw_contour_centres (canvas, valid_centres, black)
        nc = len (selected_centres) + len (valid_centres)

        # Say how many stomata are on the image.
        text = "%d" % nc
        cv2.putText (canvas, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                     1, 0, 2, cv2.LINE_AA)

        # Display the image and allow the user to type:
        #   "q" to exit the program.
        #   " " or "n" to move to the next image on the command line
        #   "s" to save the image currently being displayed to a file
        cv2.imshow (me, canvas)
        key = cv2.waitKey (1) & 0xFF
        if key  == ord ("q"):
            print (fn + ":", nc)
            exit (0)
        if key  == ord ("n") or key == ord (" "):
            break
        elif key == ord ("s"):
            dir, indirfn = os.path.split (fn)
            cv2.imwrite ("counted-" + indirfn, canvas)

    # Having exited the processing loop, output the count and stomata locations.
    print (fn + ":", nc)
    for c in selected_centres + valid_centres:
        print ("   %5d %5d" % tuple (c))

#-------------------------------------------------------------------------------
# End of count-stomata.
#-------------------------------------------------------------------------------
