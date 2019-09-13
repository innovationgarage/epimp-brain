import libjevois as jevois
import cv2
import numpy as np
import time

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class ColorTracker:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):

        def tellRobot(bbox, out_center_x, out_center_y, serial_format="XY"):
            if bbox is None:
                jevois.sendSerial("stop")
            else:
                box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                if serial_format == "XY":
                    if out_center_x < box_center_x:
                        move_x = box_center_x - out_center_x
                    elif box_center_x < out_center_x:
                        move_x = out_center_x - box_center_x
                    elif box_center_x == out_center_x:
                        move_x = 0
                    if out_center_y < box_center_y:
                        move_y = box_center_y - out_center_y
                    elif box_center_y < out_center_y:
                        move_y = out_center_y - box_center_y
                    elif box_center_y == out_center_y:
                        move_y = 0
                    if move_x < 100:
                        move_x = 100
                    if move_y < 100:
                        move_y = 100                        
                    jevois.sendSerial("smoothmove {} {}".format(int(move_x), int(move_y)))
                else:
                    jevois.sendSerial("Invalid Serial Format")

        img = inframe.getCvBGR()
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
        out_center_x, out_center_y = frameWidth/2, frameHeight/2

        # Set the frame rate
        time.sleep(0.2)

        # Set the serial output format
        serial_format = "XY" #Options: "Belts", "XY"

        # Preprocess the input
        blurred = cv2.bilateralFilter(img,9,75,75)
        # blurred = cv2.GaussianBlur(img, (21, 21), 0)
        ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
        mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

        # Setup the tracker
        tracker = cv2.TrackerKCF_create()
        bbox = None

        # Filter the desired color range
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        redLower = (0,10,70)
        redUpper = (40,255,255)
        image = cv2.inRange(hsv, redLower, redUpper)
        image = cv2.erode(image, None, iterations=2)
        image = cv2.dilate(image, None, iterations=2)

        # Find the biggest contour
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        if contours:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            x,y,w,h = cv2.boundingRect(biggest_contour)
            box_center_x, box_center_y = x+w/2, y+h/2
            cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

            # Track the biggest contour
            if bbox is None:
                bbox = (x, y, w, h)
                ok = tracker.init(img, bbox)
                cv2.rectangle(mask,(x,y), (x+w, y+h), (0,255,0), 2)                
            else:
                ok, bbox = tracker.update(img)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                    cv2.rectangle(mask,p1, p2, (0,255,0), 2)
                else:
                    bbox = None
        else:
            bbox = None
        cv2.putText(mask, "BBOX: " + str(bbox), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)

        # Tell the robot what to do
        tellRobot(bbox, out_center_x, out_center_y)

        # Organize the visual output
        toprow = np.hstack((img, blurred))
        bottomrow = np.hstack((thresh, mask))
        outimg = np.vstack((toprow, bottomrow))
        outframe.sendCv(outimg)
                
    # ###################################################################################################
    ## Process function without USB output
    def processNoUSB(self, inframe):

        def tellRobot(bbox, out_center_x, out_center_y, serial_format="XY"):
            if bbox is None:
                jevois.sendSerial("stop")
            else:
                box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                if serial_format == "XY":
                    if out_center_x < box_center_x:
                        move_x = box_center_x - out_center_x
                    elif box_center_x < out_center_x:
                        move_x = out_center_x - box_center_x
                    elif box_center_x == out_center_x:
                        move_x = 0
                    if out_center_y < box_center_y:
                        move_y = box_center_y - out_center_y
                    elif box_center_y < out_center_y:
                        move_y = out_center_y - box_center_y
                    elif box_center_y == out_center_y:
                        move_y = 0
                    if move_x < 100:
                        move_x = 100
                    if move_y < 100:
                        move_y = 100                        
                    jevois.sendSerial("smoothmove {} {}".format(int(move_x), int(move_y)))
                else:
                    jevois.sendSerial("Invalid Serial Format")

        img = inframe.getCvBGR()
        out_x, out_y = 352, 288
        out_center_x, out_center_y = out_x/2, out_y/2

        # Set the frame rate
        time.sleep(0.2)

        # Set the serial output format
        serial_format = "XY" #Options: "Belts", "XY"

        # Preprocess the input
        blurred = cv2.bilateralFilter(img,9,75,75)
        # blurred = cv2.GaussianBlur(img, (21, 21), 0)
        ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
        mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

        # Setup the tracker
        tracker = cv2.TrackerKCF_create()
        bbox = None

        # Filter the desired color range
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        redLower = (0,10,70)
        redUpper = (40,255,255)
        image = cv2.inRange(hsv, redLower, redUpper)
        image = cv2.erode(image, None, iterations=2)
        image = cv2.dilate(image, None, iterations=2)

        # Find the biggest contour
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        if contours:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            x,y,w,h = cv2.boundingRect(biggest_contour)
            box_center_x, box_center_y = x+w/2, y+h/2
            cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

            # Track the biggest contour
            if bbox is None:
                bbox = (x, y, w, h)
                ok = tracker.init(img, bbox)
                cv2.rectangle(mask,(x,y), (x+w, y+h), (0,255,0), 2)                
            else:
                ok, bbox = tracker.update(img)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
                    cv2.rectangle(mask,p1, p2, (0,255,0), 2)
                else:
                    bbox = None
        else:
            bbox = None

        # Tell the robot what to do
        tellRobot(bbox, out_center_x, out_center_y)
