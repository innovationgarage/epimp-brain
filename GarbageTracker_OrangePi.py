import cv2
import numpy as np
import sys
import time
import serial

## Object detection and recognition using OpenCV Deep Neural Networks (DNN)
#
# This module runs an object detection deep neural network using the OpenCV DNN
# library. Detection networks analyze a whole scene and produce a number of
# bounding boxes around detected objects, together with identity labels
# and confidence scores for each detected box.
#
# This module supports detection networks implemented in TensorFlow, Caffe,
# Darknet, Torch, etc as supported by the OpenCV DNN module.
#
# Included with the standard JeVois distribution are:
#
# - OpenCV Face Detector, Caffe model
# - MobileNet + SSD trained on Pascal VOC (20 object classes), Caffe model
# - MobileNet + SSD trained on Coco (80 object classes), TensorFlow model
# - MobileNet v2 + SSD trained on Coco (80 object classes), TensorFlow model
# - Darknet Tiny YOLO v3 trained on Coco (80 object classes), Darknet model
# - Darknet Tiny YOLO v2 trained on Pascal VOC (20 object classes), Darknet model
#
# See the module's constructor (__init__) code and select a value for \b model to switch network. Object categories are
# as follows:
# - The 80 COCO object categories are: person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic,
#   fire, stop, parking, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
#   handbag, tie, suitcase, frisbee, skis, snowboard, sports, kite, baseball, baseball, skateboard, surfboard, tennis,
#   bottle, wine, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot, pizza, donut,
#   cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell,
#   microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy, hair, toothbrush.
#
# - The 20 Pascal-VOC object categories are: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
#   diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.
#
# Sometimes it will make mistakes! The performance of yolov3-tiny is about 33.1% correct (mean average precision) on
# the COCO test set. The OpenCV Face Detector is quite fast and robust!
#
# This module is adapted from the sample OpenCV code:
# https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
#
# More pre-trained models are available on github in opencv_extra
#
#
# @author Laurent Itti
# 
# @videomapping YUYV 640 502 20.0 YUYV 640 480 20.0 JeVois PyDetectionDNN
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class GarbageTracker(object):
    # ####################################################################################################
    ## Constructor
    def __init__(self):

        def openSerialPort():
            ser = serial.serial_for_url('/dev/ttyUSB0')
            ser = serial.Serial('/dev/ttyACM0')
            ser.baudrate = 115200
            ser.timeout = 1
            return ser
        
        try:
            self.ser = openSerialPort()
        except:
            self.ser = None
            print('Opening serial failed')

        self.confThreshold = 0.5 # Confidence threshold (0..1), higher for stricter detection confidence.
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold (0..1), higher to remove more duplicate boxes.
        self.inpWidth = 160      # Resized image width passed to network
        self.inpHeight = 120     # Resized image height passed to network
        self.scale = 2/255       # Value scaling factor applied to input pixels
        self.mean = [127.5, 127.5, 127.5] # Mean BGR value subtracted from input image
        self.rgb = True          # True if model expects RGB inputs, otherwise it expects BGR
        self.bbox = None
#        self.tracker = cv2.TrackerKCF_create()

        # Select one of the models:
        #model = 'Face'              # OpenCV Face Detector, Caffe model
        #model = 'MobileNetV2SSD'   # MobileNet v2 + SSD trained on Coco (80 object classes), TensorFlow model
        #model = 'MobileNetSSD'     # MobileNet + SSD trained on Pascal VOC (20 object classes), Caffe model
        #model = 'MobileNetSSDcoco' # MobileNet + SSD trained on Coco (80 object classes), TensorFlow model
        #model = 'YOLOv3'           # Darknet Tiny YOLO v3 trained on Coco (80 object classes), Darknet model
        #model = 'YOLOv2'           # Darknet Tiny YOLO v2 trained on Pascal VOC (20 object classes), Darknet model
        model = 'RPi'

        #model = 'MobileNetSSDcoco' # MobileNet + SSD trained on Coco (80 object classes), TensorFlow model        
        #model = 'RPi'
        model = 'FasterRCNN'

        # You should not have to edit anything beyond this point.
        backend = cv2.dnn.DNN_BACKEND_DEFAULT
        target = cv2.dnn.DNN_TARGET_CPU
        self.classes = None
        classnames = None
        if (model == 'MobileNetSSD'):
            classnames = '/jevois/share/darknet/yolo/data/voc.names'
            modelname = '/jevois/share/opencv-dnn/detection/MobileNetSSD_deploy.caffemodel'
            configname = '/jevois/share/opencv-dnn/detection/MobileNetSSD_deploy.prototxt'
            self.rgb = False
        elif (model == 'MobileNetV2SSD'):
            classnames = '/jevois/share/darknet/yolo/data/coco.names'
            modelname = '/jevois/share/opencv-dnn/detection/ssd_mobilenet_v2_coco_2018_03_29.pb'
            configname = '/jevois/share/opencv-dnn/detection/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
        elif (model == 'MobileNetSSDcoco'):
            classnames = 'TFModels/coconew.names'
            modelname = 'TFModels/ssd_mobilenet_v1_coco_2017_11_17.pb'
            configname = 'TFModels/ssd_mobilenet_v1_coco_2017_11_17.pbtxt'
            self.rgb = False
            self.nmsThreshold = 0.1
        elif (model == 'RPi'):
            classnames = 'TFModels/coconew.names'
            modelname = 'TFModels/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'
            configname = 'TFModels/ssd_mobilenet_v1_ppn_coco.pbtxt'
            self.rgb = False
            self.nmsThreshold = 0.1
        elif (model == 'FasterRCNN'):
            classnames = 'TFModels/coconew.names'
            modelname = 'TFModels/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
            configname = 'TFModels/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'
            self.rgb = False
            self.nmsThreshold = 0.1
        elif (model == 'YOLOv3'):
            classnames = '/jevois/share/darknet/yolo/data/coco.names'
            modelname = '/jevois/share/darknet/yolo/weights/yolov3-tiny.weights'
            configname = '/jevois/share/darknet/yolo/cfg/yolov3-tiny.cfg'
        elif (model == 'YOLOv2'):
            classnames = '/jevois/share/darknet/yolo/data/voc.names'
            modelname = '/jevois/share/darknet/yolo/weights/yolov2-tiny-voc.weights'
            configname = '/jevois/share/darknet/yolo/cfg/yolov2-tiny-voc.cfg'
            self.inpWidth = 320
            self.inpHeight = 240
        else:
            classnames = '/jevois/share/opencv-dnn/detection/opencv_face_detector.classes'
            modelname = '/jevois/share/opencv-dnn/detection/opencv_face_detector.caffemodel'
            configname = '/jevois/share/opencv-dnn/detection/opencv_face_detector.prototxt'
            self.scale = 1.0
            self.mean = [104.0, 177.0, 123.0]
            self.rgb = False

        # Load names of classes
        if classnames:
            with open(classnames, 'rt') as f:
                self.classes = f.read().rstrip('\n').split('\n')
        
        # Load a network
        self.net = cv2.dnn.readNet(modelname, configname)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self.model = model
        garbageclasses = ["shoe", "hat", "eye glasses", "frisbee",
            "bottle", "plate", "wine glass", "cup", "fork", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "fruit",
            "hotdog", "pizza", "donut", "cake",
            "vase", "scissors", "toothbrush", "cardboard", "napkin",
            "net", "paper", "plastic", "straw"]
        self.garbageclasses = garbageclasses
        
    # ####################################################################################################
    ## Get names of the network's output layers
    def getOutputsNames(self, net):
        layersNames = self.net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # ####################################################################################################
    ## Analyze and draw boxes, object names, and confidence scores
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        out_center_x, out_center_y = frameWidth/2, frameHeight/2
        
    def sendSerial(ser, msg):
        if not ser.is_open:
            print('Serial port is not open!')
            return False
        else:
            msg = "{}\n".format(msg)
            print(msg)
            ser.write(msg.encode())

        def track(classId, conf, box):
            # Track the last box on the list
            if self.bbox is None:
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                self.bbox = (left, top, left + width, top + height)
                ok = self.tracker.init(frame, self.bbox)
                drawPred(classId, conf, left, top, left + width, top + height, color=(255, 255, 255))
            else:
                ok, self.bbox = self.tracker.update(frame)
                if ok:
                    p1 = (int(self.bbox[0]), int(self.bbox[1]))
                    p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                    drawPred(classId, conf, p1[0], p1[1], p2[0], p2[1], color=(255, 0, 0))
                else:
                    self.bbox = None

        def drawPred(classId, conf, left, top, right, bottom, color=(0,255,0)):
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = '%.2f' % (conf * 100)

            # Print a label of class.
            if self.classes:
                if (classId >= len(self.classes)):
                    label = 'Oooops id=%d: %s' % (classId, label)
                else:
                    label = '%s: %s' % (self.classes[classId], label)
                        
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (left, top - labelSize[1]-2), (left + labelSize[0], top + baseLine),
                         (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

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
                        
        layerNames = self.net.getLayerNames()
        lastLayerId = self.net.getLayerId(layerNames[-1])
        lastLayer = self.net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        if self.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for out in outs:
                for detection in out[0, 0]:
                    classId = int(detection[1]) - 1
                    confidence = detection[2]
                    is_garbage = self.classes[classId] in self.garbageclasses
                    if (confidence > self.confThreshold):# and (is_garbage):
                        left = int(detection[3])
                        top = int(detection[4])
                        right = int(detection[5])
                        bottom = int(detection[6])
                        width = right - left + 1
                        height = bottom - top + 1
                        classIds.append(classId)  # Skip background label
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
                            
        elif lastLayer.type == 'DetectionOutput':
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for out in outs:
                for detection in out[0, 0]:
                    classId = int(detection[1]) - 1
                    confidence = detection[2]
                    is_garbage = self.classes[classId] in self.garbageclasses
                    if (confidence > self.confThreshold):# and (is_garbage):
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                        classIds.append(classId)  # Skip background label
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        elif lastLayer.type == 'Region':
            # Network produces output blob with a shape NxC where N is a number of
            # detected objects and C is a number of classes + 4 where the first 4
            # numbers are [center_x, center_y, width, height]
            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    is_garbage = self.classes[classId] in self.garbageclasses
                    if (confidence > self.confThreshold):# and (is_garbage):
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        else:
            print('Unknown output layer type: ' + lastLayer.type)
            return

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        print('HERE', indices)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, color=(0, 255, 0))
            # track(classIds[i], confidences[i], box)
            # Tell the robot what to do
            if ser:
                tellRobot(self.ser, self.bbox, frameWidth, frameHeight)
            else:
                tellMe(self.bbox, frameWidth, frameHeight)
            

    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe):
        frame = inframe
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        out_center_x, out_center_y = frameWidth/2, frameHeight/2

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, self.scale, (self.inpWidth, self.inpHeight), self.mean, self.rgb, crop=False)

        # Run a model
        self.net.setInput(blob)
        if self.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
            frame = cv2.resize(frame, (self.inpWidth, self.inpHeight))
            self.net.setInput(np.array([self.inpHeight, self.inpWidth, 1.6], dtype=np.float32), 'im_info')
        outs = self.net.forward(self.getOutputsNames(self.net))
        
        self.postprocess(frame, outs)

        # Create dark-gray (value 80) image for the bottom panel, 22 pixels tall:
        msgbox = np.zeros((22, frame.shape[1], 3), dtype = np.uint8) + 80
      
        # Put efficiency information.
        cv2.putText(frame, 'JeVois Python Object Detection DNN - ' + self.model, (3, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        t, _ = self.net.getPerfProfile()

#        label = fps + ' - Inference time: %.2fms' % (t * 1000.0 / cv2.getTickFrequency())
        label = 'bbox: {}'.format(self.bbox)
        cv2.putText(msgbox, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Stack bottom panel below main image:
        outframe = np.vstack((frame, msgbox))

        return outframe


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)
    return cap

def main():    
    od = GarbageTracker()
    cap = cv2.VideoCapture(0)
    cap = change_res(cap, 320, 240)    
    time.sleep(0.2)
    bbox = None
    while True:
        time.sleep(0.5)
        _, inframe = cap.read()
        if inframe is None:
            break

        frameWidth, frameHeight = inframe.shape[0], inframe.shape[1]
        outframe = od.process(inframe)
        cv2.imshow("outimg", outframe)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    
if __name__=="__main__":
    main()
