import cv2
import numpy as np
import time

tracker = cv2.TrackerKCF_create()
vs = cv2.VideoCapture(0)
time.sleep(1)

bbox = None
while True:
    time.sleep(0.2)
    _, img = vs.read()
    # cv2.imshow("img", img)
    if img is None:
        break

    blurred = cv2.bilateralFilter(img,9,75,75)
#    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
    mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)

    redLower = (0,10,70)
    redUpper = (40,255,255)

    redMask = cv2.inRange(hsv, redLower, redUpper)
    redMask = cv2.erode(redMask, None, iterations=2)
    redMask = cv2.dilate(redMask, None, iterations=2)
    
    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, None, iterations=2)
    greenMask = cv2.dilate(greenMask, None, iterations=2)

    target = cv2.bitwise_or(redMask, greenMask)
    target = greenMask
#    target = cv2.bitwise_and(img,img, mask=colorMask)
    
    image = cv2.erode(target, None, iterations=2)
    image = cv2.dilate(target, None, iterations=2)

    # Find the biggest contour
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if contours:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,w,h = cv2.boundingRect(biggest_contour)
        box_center_x, box_center_y = x+w/2, y+h/2
        #cv2.rectangle(mask, (x,y), (x+w,y+h), (0,255,0), 2)
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
                cv2.rectangle(mask,p1, p2, (0,255,0), 2)
            else:
                bbox = None
    else:
        bbox = None

    cv2.putText(mask, "current BBOX: " + str(bbox), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
#    cv2.imshow("img", mask)
    
    toprow = np.hstack((img, blurred))
    bottomrow = np.hstack((thresh, mask))
    outimg = np.vstack((toprow, bottomrow))
    cv2.imshow("outimg", outimg)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    
