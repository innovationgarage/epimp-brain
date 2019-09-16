from threading import Thread
import cv2
#import prctl
DEBUG = False

class ObjectTrack:
    """
    Class that tracks the bbos it receives
    """
    
    def __init__(self, getter=None, detector=None, resize_factor=1, grayscale=False):
        self.getter = getter
        self.detector = detector
        self.stopped = False
        self.bbox = None
        self.previous_detection = None
        self.resize_factor = resize_factor
        self.grayscale = grayscale
        
    def start(self):
        Thread(target=self.track, args=()).start()
        return self

    def preProcess(self, frame):
        frame = cv2.resize(frame, (int(frame.shape[1] / self.resize_factor), int(frame.shape[0] / self.resize_factor)))
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
        
    def track(self):
        # prctl.set_name('ObjectTrack')
        while not self.stopped:
            detection, history = self.detector.top_detection
            getterframe_all = self.getter.frame
            if getterframe_all is None:
                continue
            getterframe_original = getterframe_all[1]
            getterframe = self.preProcess(getterframe_original)
            cols, rows = getterframe.shape[0], getterframe.shape[1]
            #track the detected object
            if (detection is not None) and (detection is not self.previous_detection):
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                width = int(right - left)
                height = int(bottom - top)
                self.bbox = (left, top, right, bottom)
                self.tracker = cv2.TrackerMedianFlow_create()
                tracker_bbox = (left, top, width, height)
                tracker_bbox = tuple([int(c/self.resize_factor) for c in tracker_bbox])
                ok = self.tracker.init(self.preProcess(history[0][1]), tracker_bbox)
                if DEBUG:  print('Detection', ok, self.bbox)
                for frame_tuple in history:
                    frame = self.preProcess(frame_tuple[1])
                    ok, self.bbox = self.tracker.update(frame)
                    if DEBUG:  print('History', ok, self.bbox)
                self.previous_detection = detection
            else:
                if self.previous_detection is not None:
                    ok, tracker_bbox = self.tracker.update(getterframe)
                    bbox = [int(c*self.resize_factor) for c in tracker_bbox]
                    (left, top, width, height) = bbox
                    self.bbox = (left, top, left+width, top+height)
                    if DEBUG:  print('Camera', ok, self.bbox)
                else:
                    self.bbox = None
                    
    def stop(self):
        self.stopped = True
