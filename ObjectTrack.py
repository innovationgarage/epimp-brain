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
            getterframe = getterframe_all[1]
            cols, rows = getterframe.shape[0], getterframe.shape[1]
            scaled_getterframe = self.preProcess(getterframe)
            scaled_cols, scaled_rows = scaled_getterframe.shape[0], scaled_getterframe.shape[1]
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
                scaled_tracker_bbox = tuple([int(c/self.resize_factor) for c in tracker_bbox])
                ok = self.tracker.init(self.preProcess(history[0][1]), scaled_tracker_bbox)
                if DEBUG:  print('Detection', ok, self.bbox)
                for frame_tuple in history:
                    scaled_frame = self.preProcess(frame_tuple[1])
                    ok, scaled_bbox = self.tracker.update(scaled_frame)
                    self.bbox = tuple([int(c*self.resize_factor) for c in scaled_bbox])
                    if DEBUG:  print('History', ok, self.bbox)
                self.previous_detection = detection
            else:
                if self.previous_detection is not None:
                    ok, scaled_tracker_bbox = self.tracker.update(scaled_getterframe)
                    bbox = [int(c*self.resize_factor) for c in scaled_tracker_bbox]
                    (left, top, width, height) = bbox
                    self.bbox = (left, top, left+width, top+height)
                    if DEBUG:  print('Camera', ok, self.bbox)
                else:
                    self.bbox = None
                    
    def stop(self):
        self.stopped = True
