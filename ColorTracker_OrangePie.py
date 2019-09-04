import cv2
import numpy as np
import time
import serial

def openSerialPort():
#    ser = serial.serial_for_url('/dev/ttyS1')
    ser = serial.Serial('/dev/ttyACM0')
    ser.baudrate = 74880
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
        
def tellRobot(ser, bbox, frameWidth, frameHeight, serial_format="XY"):
    if bbox is None:
        sendSerial(ser, "stop")
    else:
        box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        out_center_x, out_center_y = frameWidth/2., frameHeight/2.
        move_x = -(out_center_x - box_center_x)/(frameWidth/2.)*100.
        move_y = (out_center_y - box_center_y)/(frameHeight/2.)*100.
        sendSerial(ser, "move {} {} 1000".format(int(move_x), int(move_y)))

def main():
    ser = openSerialPort()
    tracker = cv2.TrackerKCF_create()
    vs = cv2.VideoCapture(0)
    time.sleep(.2)

    bbox = None
    while True:
        time.sleep(0.2)
        _, img = vs.read()
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
        tracker = cv2.TrackerKCF_create()
        bbox = None
        
        # Filter the desired color range
        redLower = (0,10,10)
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
            #cv2.rectangle(mask, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

            # Track the biggest contour
            if bbox is None:
                bbox = (x, y, w, h)
                ok = tracker.init(img, bbox)
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

        cv2.putText(mask, "current BBOX: " + str(bbox), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Tell the robot what to do
        tellRobot(ser, bbox, frameWidth, frameHeight)

        # Organize the visual output
        toprow = np.hstack((img, blurred))
        bottomrow = np.hstack((thresh, mask))
        outimg = np.vstack((toprow, bottomrow))
        cv2.imshow("outimg", outimg)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    
if __name__=="__main__":
    main()
