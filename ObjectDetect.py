from threading import Thread
import cv2
import numpy as np
import prctl

class ObjectDetect:
    """
    Class tat continuously looks for objects in garbage classes
    """
    def __init__(self, cvNet, threshold, classes, garbageclasses, getter=None):
        self.getter = getter
        self.stopped = False
        self.cvNet = cvNet
        self.threshold = threshold
        self.classes = classes
        self.garbageclasses = garbageclasses
        self.detectionno = 0
        self.top_detection = None

    def start(self):
        Thread(target=self.detect, args=()).start()
        return self

    def detect(self):
        prctl.set_name('ObjectDetect')
        while not self.stopped:
            self.detectionno += 1
            self.frame = self.getter.frame
            if self.frame:
                self.cvNet.setInput(cv2.dnn.blobFromImage(self.frame[1], size=(300, 300), swapRB=True, crop=False))
                cvOut = self.cvNet.forward()
                detections = [d for d in cvOut[0,0,:,:] if float(d[2])>self.threshold and self.classes[int(d[1])] in self.garbageclasses]
                scores = [d[2] for d in cvOut[0,0,:,:] if float(d[2])>self.threshold and self.classes[int(d[1])] in self.garbageclasses]
                if len(scores)>0:
                    self.top_detection = detections[np.argmax(scores)]
                else:
                    self.top_detection = None

    def stop(self):
        self.stopped = True
