from threading import Thread
import cv2
import numpy as np
#import prctl
import time

class ObjectDetect:
    """
    Class that continuously looks for objects in garbage classes
    """
    def __init__(self, cvNet, threshold, classes, garbageclasses, getter=None):
        self.getter = getter
        self.stopped = False
        self.cvNet = cvNet
        self.threshold = threshold
        self.classes = classes
        self.garbageclasses = garbageclasses
        self.detectionno = 0
        self.top_detection = None, None
        self.frame = None

    def start(self):
        Thread(target=self.detect, args=()).start()
        return self

    def detect(self):
        # prctl.set_name('ObjectDetect')
        while not self.stopped:
            self.detectionno += 1
            if not self.frame:
                self.frame = self.getter.frame
                continue
                        
            t0 = time.time()
            self.cvNet.setInput(cv2.dnn.blobFromImage(self.frame[1], size=(300, 300), swapRB=True, crop=False))
            cvOut = self.cvNet.forward()
            time.sleep(1.)
            detections = [d for d in cvOut[0,0,:,:] if float(d[2])>self.threshold and self.classes[int(d[1])] in self.garbageclasses]
            scores = [d[2] for d in cvOut[0,0,:,:] if float(d[2])>self.threshold and self.classes[int(d[1])] in self.garbageclasses]
            getterrecord = self.getter.record()
            self.frame = self.getter.frames[0]
            if len(scores)>0:
                t1 = time.time()
                if len(getterrecord)==0:
                    print('Detection took', t1-t0)
                self.top_detection = detections[np.argmax(scores)], getterrecord
            else:
                self.top_detection = None, None

    def stop(self):
        self.stopped = True
