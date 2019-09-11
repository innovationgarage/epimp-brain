import cv2
import time
import numpy as np
import serial

def openSerialPort():
    ser = serial.Serial('/dev/ttyUSB0')
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

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)
    return cap

def scale(x, maxval):
    if x == 0:
        return 0
    elif x<0:
        res = np.log(-x)
        return -(res/maxval*100)
    elif x>0:
        res = np.log(x)
        return res/maxval*100

def tellMe(bbox, frameWidth, frameHeight, serial_format="XY"):
    if bbox is None:
        return "move -x 0 -y 0 1000"
    else:
        box_center_x, box_center_y = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        out_center_x, out_center_y = frameWidth/2., frameHeight/2.
        move_x = (out_center_x - box_center_x)
        move_y = -(out_center_y - box_center_y)
        move_x = scale(move_x, np.log(frameWidth/2.))
        move_y = scale(move_y, np.log(frameHeight/2.))
        return "move -x {} -y {} 1000".format(int(move_x), int(move_y))

def tellRobot(ser, bbox, frameWidth, frameHeight, serial_format="XY"):
    sendSerial(ser, tellMe(bbox, frameWidth, frameHeight, serial_format).encode())

def drawPred(img, bbox, color=(0,255,0), label=None):
    if bbox is not None:
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])
        # Draw a bounding box.
        cv2.rectangle(img, (left, top), (right, bottom), color, 5)
        # Print a label of class.
        if label is not None:                        
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img

def lookForGarbage(img, cvNet, threshold, classes, garbageclasses, draw=False):
    rows = img.shape[0]
    cols = img.shape[1]

    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    detections = [d for d in cvOut[0,0,:,:] if float(d[2])>threshold and classes[int(d[1])] in garbageclasses]
    scores = [d[2] for d in cvOut[0,0,:,:] if float(d[2])>threshold and classes[int(d[1])] in garbageclasses]
    if len(scores)>0:
        top_detection = detections[np.argmax(scores)]
        score = float(top_detection[2])
        classId = int(top_detection[1])
        left = top_detection[3] * cols
        top = top_detection[4] * rows
        right = top_detection[5] * cols
        bottom = top_detection[6] * rows
        width = right - left
        bbox = (left, top, right, bottom)
        if draw:
            label = '{} {}'.format(classes[classId], np.round(top_detection[2]*100., 2))
            img = drawPred(img, bbox, color=(23,230,210), label=label)
    else:
        bbox = None
    return img, bbox

def track(img, tracker, bbox):
    if tracker is None:
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(img, bbox)
    else:
        ok, bbox = tracker.update(img)
        if not ok:
            tracker = None
            bbox = None
    return bbox, tracker
            
def main(model, threshold):
    if model == 'TF':
        modelname = 'TFModels/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'
        configname = 'TFModels/ssd_mobilenet_v1_ppn_coco.pbtxt'
        classnames = 'TFModels/coconew.names'
    elif model == 'DarkNet':
        modelname = 'TFModels/yolov2-tiny.weights'
        configname = 'TFModels/yolov2-tiny.cfg'
        classnames = 'TFModels/coconew.names'
    garbageclasses = [
#        "person",
        "shoe", "hat", "eye glasses", "frisbee",
        "bottle", "plate", "wine glass", "cup", "fork", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "fruit",
        "hotdog", "pizza", "donut", "cake",
        "vase", "scissors", "toothbrush", "cardboard", "napkin",
        "net", "paper", "plastic", "straw"
    ]
    bbox = None
    tracker = None
    
    with open(classnames, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    if model == 'TF':
        cvNet = cv2.dnn.readNetFromTensorflow(modelname, configname)
    elif model == 'DarkNet':
        cvNet = cv2.dnn.readNetFromDarknet(configname, modelname)

    cap = cv2.VideoCapture(0)
    cap = change_res(cap, 320, 240)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(fps)
    time.sleep(0.2)

    try:
        ser = openSerialPort()
    except:
        ser = None
        print('Opening serial failed')

    start = time.time()
    framecount = 0
    while True:
        _, img = cap.read()
        framecount += 1

        if img is None:
            break
        rows = img.shape[0]
        cols = img.shape[1]

        if framecount%60==0:
            end = time.time()
            time_elapsed = end - start
            img, bbox = lookForGarbage(img, cvNet, threshold, classes, garbageclasses, draw=True)
            cv2.putText(img, 'fps: {}'.format(int(framecount/time_elapsed)), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('detect', img)
        else:
            bbox, tracker = track(img, tracker, bbox)
            end = time.time()
            time_elapsed = end - start
            cv2.putText(img, 'fps: {}'.format(int(framecount/time_elapsed)), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            img = drawPred(img, bbox, color=(255,0,0))
            cv2.imshow('track', img)
            
        resized = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)    
        cv2.imshow('img', img)
    
        # Tell the robot what to do
        if ser:
            tellRobot(ser, bbox, cols, rows)
            print('fps: {}'.format(framecount/time_elapsed))
        else:
            print(tracker, bbox)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__=="__main__":
    main(model='TF', threshold=0.5)
