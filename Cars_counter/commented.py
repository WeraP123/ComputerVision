#!/usr/bin/env python3
"""
This program is adapted from the tutorial at
https://www.learnpythonwithrune.org/opencv-counting-cars-a-simple-approach/
"""

#-------------------------------------------------------------------------------
# Boilerplate.
#-------------------------------------------------------------------------------
import sys, argparse
import cv2

# Set up the version from the timestamp which Emacs automagically updates.
version = \
    "Time-stamp: <2022-12-05 17:33:26 Adrian F Clark (alien@essex.ac.uk)>"
version = version[13:33]

#-------------------------------------------------------------------------------
# Routines.
#-------------------------------------------------------------------------------

def in_range (x, lo, hi):
    return x >= lo and x <= hi

def load_config (fn):
    "Read a configuration file and return the configuration."

    # Each region consists its the upper left and lower right coordinates.
    # All are integers.
    regions = []
    with open (fn) as f:
        for line in f:
            line = line[:-1]
            if len (line) == 0: continue
            if line[0] == "#": continue
            words = line.split ()
            if len (words) != 4:
                print ("Invalid configuration line: '%s'" % line,
                       file=sys.stderr)
                exit (1)
            vals = []
            for w in words:
                try:
                    v = int (w)
                except:
                    print ("Invalid integer: '%s'" % w, file=sys.stderr)
                    exit (2)
                vals += [v]
            vals += [0]  # last element records the next frame to consider
            regions += [vals]

    # Return the configuration.
    return regions

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

parser = argparse.ArgumentParser (description="Count cars in a video.")

parser.add_argument ("-c", "--config", default=None,
                     help="configuration for this camera or video")
parser.add_argument ("-t", "--threshold", type=int, default=25,
                     help="threshold for binarizing the image")
parser.add_argument ("-s", "--contoursize", type=int, default=500,
                     help="minimum size of a car contour")
parser.add_argument ("-d", "--deadframes", type=int, default=5,
                     help="number of frames to ignore after detecting a car")
parser.add_argument ("--delay", type=int, default=1,
                     help="number of ms to display each frame")
parser.add_argument ("-v", "--version", action="store_true", default=False,
                     help="output the program version")
parser.add_argument ("-debug", action="store_true", default=False,
                     help="output stuff while running")

parser.add_argument ('video', nargs='?', default='video0')

args = parser.parse_args()

# Output the version if requested.
if args.version:
    print (version)

# Handle the command line.  If no video file was supplied, use the first camera
# attached to the system, in which case we need a configuration file.  If a
# video file was supplied, base the configuration filename on it.
if args.video == "video0":
    if args.config is None:
        print ("You need to specify a config file!", file=sys.stderr)
        exit (99)
    else:
        regions = load_config (args.config)
    cap = cv2.VideoCapture (0)
else:
    if args.config is None:
        regions = load_config (args.video[:-3] + "cfg")
    else:
        regions = load_config (args.config)
    cap = cv2.VideoCapture (args.video)

# The title of the window we display.
title = " ".join (sys.argv)

# We retain the previous frame to see if there has been any movement.
prev_frame = None

# We use a global frame counter to help us avoid multiple detections.
frame_number = 0

# The number of cars counted.
car_count = 0

# Loop over all the frames.
while cap.isOpened ():
    _, frame = cap.read ()
    if frame is None: break
    frame_number += 1

    # Prepare the image:
    #   -- convert to grey-scale
    #   -- blur so small changes are ignored
    grey = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur (grey, (5, 5), 0)

    # On the first iteration, initialize things for subsequent processing.
    if prev_frame is None or prev_frame.shape != grey.shape:
        prev_frame = grey
        continue

    # Compute the absolute difference from the previous frame, then update
    # prev_frame.
    delta_frame = cv2.absdiff (prev_frame, grey)
    prev_frame = grey

    # Threshold the image, dilate it a bit, then find contours.
    thresh = cv2.threshold (delta_frame, args.threshold, 255,
                            cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate (thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # Work through the contours found, ignoring small ones.
    for contour in contours:
        if cv2.contourArea(contour) < args.contoursize:
            continue

        # Mark the bounding box on the image.
        x, y, w, h = cv2.boundingRect (contour)
        cv2.rectangle (frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        xmid = x + w //2

        # Work through all the pre-defined regions.
        for reg in regions:
            colour = (0, 0, 0) if reg[-1] > frame_number else (0, 0, 255)
            cv2.rectangle(frame, (reg[0], reg[1]), (reg[2], reg[3]),
                          colour, 2)
            cv2.rectangle(frame, (xmid, y+h), (xmid+1, y+h-1), (0,0,255), 2)

            # Do nothing if we are in the dead time after detecting a car.
            if reg[-1] > frame_number:
                continue

            # Have we found a car?
            if in_range (xmid, reg[0], reg[2]) and \
               in_range (y+h, reg[1], reg[3]):
                car_count += 1
                reg[-1] = frame_number + args.deadframes

    # Annotate the image with the text we have built up and the boxes we use
    # for counting.
    cv2.putText(frame, "Cars: %d" % car_count,
                (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Let the user see what we have done.
    cv2.imshow(title, frame)
    if cv2.waitKey(args.delay) & 0xFF == ord('q'):
        break

# Tidy up OpenCV.
cap.release()
cv2.destroyAllWindows()

# Local Variables:
# time-stamp-line-limit: 50
# End:
#-------------------------------------------------------------------------------
# End of countcars.
#-------------------------------------------------------------------------------
