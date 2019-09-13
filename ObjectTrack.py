from threading import Thread
import cv2
import prctl

class ObjectTrack:
    """
    Class that tracks the bbos it receives
    """
    
    def __init__(self, getter=None, detector=None):
        self.getter = getter
        self.detector = detector
        self.stopped = False
        self.bbox = None
        self.previous_detection = None
        
    def start(self):
        Thread(target=self.track, args=()).start()
        return self

    def track(self):
        prctl.set_name('ObjectTrack')
        while not self.stopped:
            detection, history = self.detector.top_detection
            getterframe = self.getter.frame
            if getterframe is None:
                continue
            cols, rows = getterframe[1].shape[0], getterframe[1].shape[1]
            #track the detected object
            if (detection is not None) and (detection is not self.previous_detection):
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                self.bbox = (left, top, right, bottom)
                self.tracker = cv2.TrackerKCF_create()
                ok = self.tracker.init(history[0][1], self.bbox)
                print('Detection', ok, self.bbox)
                # for frame in history:
                #     ok, self.bbox = self.tracker.update(frame[1])
                #     print('History', ok, self.bbox)
                self.previous_detection = detection
            else:
                if self.previous_detection is not None:
                    ok, self.bbox = self.tracker.update(getterframe[1])
                    print('Camera', ok, self.bbox)
                else:
                    self.bbox = None
                    
    def stop(self):
        self.stopped = True
