import cv2
import numpy as np
import time
import serial

def openSerialPort():
    ser = serial.serial_for_url('/dev/ttyUSB0')
#    ser = serial.Serial('/dev/ttyACM0')
    ser.baudrate = 115200
    ser.timeout = 1
    return ser

def sendSerial(ser, msg):
    if not ser.is_open:
        print('Serial port is not open!')
        return False
    else:
        msg = "{}\n".format(msg)
        print(msg)
        ser.write(msg.encode())

def scale(x, maxval):
    if x == 0:
        return 0
    elif x<0:
        res = np.log(-x)
        return -(res/maxval*100)
    elif x>0:
        res = np.log(x)
        return res/maxval*100
        
def tellRobot(ser, bbox, frameWidth, frameHeight, serial_format="XY"):
    if bbox is None:
        sendSerial(ser, "move -x 0 -y 0 1000")
    else:
        box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        out_center_x, out_center_y = frameWidth/2., frameHeight/2.
        move_x = (out_center_x - box_center_x)
        move_y = -(out_center_y - box_center_y)
        move_x = scale(move_x, np.log(frameWidth/2.))
        move_y = scale(move_y, np.log(frameHeight/2.))
        sendSerial(ser, "move -x {} -y {} 1000".format(int(move_x), int(move_y)))

def tellMe(bbox, frameWidth, frameHeight, serial_format="XY"):
    if bbox is None:
        print("move -x 0 -y 0 1000")
    else:
        box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        out_center_x, out_center_y = frameWidth/2., frameHeight/2.
        move_x = (out_center_x - box_center_x)
        move_y = -(out_center_y - box_center_y)
        move_x = scale(move_x, np.log(frameWidth/2.))
        move_y = scale(move_y, np.log(frameHeight/2.))
        print("move -x {} -y {} 1000".format(int(move_x), int(move_y)))

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)
    return cap

def main():
    try:
        ser = openSerialPort()
    except:
        ser = None
        print('Opening serial failed')
        
    cap = cv2.VideoCapture(0)
    cap = change_res(cap, 320, 240)
    #tracker = cv2.TrackerKCF_create()
    time.sleep(0.2)

    bbox = None
    while True:
        #time.sleep(0.1)
        _, img = cap.read()
        if img is None:
            break

        frameWidth, frameHeight = img.shape[0], img.shape[1]

        # Preprocess the input
        blurred = cv2.bilateralFilter(img,9,75,75)
        #blurred = cv2.GaussianBlur(img, (21, 21), 0)
        ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
        mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        
        # Setup the tracker
        bbox = None
        
        # # Filter the desired color range
        # redLower = (0,10,70)
        # redUpper = (40,255,255)
        # image = cv2.inRange(hsv, redLower, redUpper)
        # image = cv2.erode(image, None, iterations=2)
        # image = cv2.dilate(image, None, iterations=2)

        #definig the range of red color
        redLower=np.array([0,10,70],np.uint8)
        redUpper=np.array([30,255,255],np.uint8)

        #defining the Range of Blue color
        blueLower=np.array([160,10,70],np.uint8)
        blueUpper=np.array([180,255,255],np.uint8)
	
        #defining the Range of yellow color
        yellowLower=np.array([22,60,200],np.uint8)
        yellowUpper=np.array([60,255,255],np.uint8)
        
        redMask = cv2.inRange(hsv, redLower, redUpper)
        redMask = cv2.erode(redMask, None, iterations=2)
        redMask = cv2.dilate(redMask, None, iterations=2)
        
        yellowMask = cv2.inRange(hsv, yellowLower, yellowUpper)
        yellowMask = cv2.erode(yellowMask, None, iterations=2)
        yellowMask = cv2.dilate(yellowMask, None, iterations=2)
        
        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, None, iterations=2)
        blueMask = cv2.dilate(blueMask, None, iterations=2)
        
        RYMask = cv2.bitwise_or(redMask, yellowMask)
        RBMask = cv2.bitwise_or(redMask, blueMask)
        
        target = RBMask
        
        image = cv2.erode(target, None, iterations=2)
        image = cv2.dilate(target, None, iterations=2)

        # Find the biggest contour
        _, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        if contours:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            x,y,w,h = cv2.boundingRect(biggest_contour)
            if (w>10) and (h>10):
                box_center_x, box_center_y = x+w/2, y+h/2
                #cv2.rectangle(mask, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

                # Track the biggest contour
                if bbox is None:
                    bbox = (x, y, w, h)
                    #ok = tracker.init(img, bbox)
                    cv2.rectangle(mask,(x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.rectangle(blurred,(x,y), (x+w, y+h), (0,255,0), 2)
                else:
                    ok, bbox = tracker.update(img)
                    if ok:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(mask,p1, p2, (0,255,0), 2)
                        cv2.rectangle(blurred,p1, p2, (0,255,0), 2)
                    else:
                        bbox = None
            else:
                bbox = None
        else:
            bbox = None

        cv2.putText(mask, "current BBOX: " + str(bbox), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (50,170,50), 2)

        # Tell the robot what to do
        if ser:
            tellRobot(ser, bbox, frameWidth, frameHeight)
        else:
            tellMe(bbox, frameWidth, frameHeight)
        # Organize the visual output
        toprow = np.hstack((img, blurred))
        bottomrow = np.hstack((thresh, mask))
        outimg = np.vstack((toprow, bottomrow))

        onerow = np.hstack((blurred, mask))
        cv2.imshow("outimg", onerow)

        # cv2.namedWindow("outimg", cv2.WINDOW_NORMAL)
        # scale_percentage = 10
        # width = int(onerow.shape[1] * scale_percentage / 100)
        # height = int(onerow.shape[0] * scale_percentage / 100)
        # dim = (width, height)
        # resized = cv2.resize(onerow, dim, interpolation=cv2.INTER_AREA) 
        # cv2.imshow("outimg", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    
if __name__=="__main__":
    main()
