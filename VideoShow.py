from threading import Thread
import cv2
import numpy as np
import time
#import prctl

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, getter=None, detector=None, tracker=None):
        self.getter = getter
        self.detector = detector
        self.tracker = tracker
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        # prctl.set_name('VideoShow')
        while not self.stopped:
            time.sleep(0.1)
            frame = self.getter.frame
            if frame:
                cols, rows = frame[1].shape[0], frame[1].shape[1]
                detection, history = self.detector.top_detection
                if detection is not None:
                    score = float(detection[2])
                    classId = int(detection[1])
                    left = int(detection[3] * cols)
                    top = int(detection[4] * rows)
                    right = int(detection[5] * cols)
                    bottom = int(detection[6] * rows)
#                    detector_bbox = (left, top, right, bottom)                    
                    cv2.rectangle(frame[1], (left, top), (right, bottom), (0,0,255), 5)
                    label = '{} {}'.format(self.detector.classes[classId], np.round(detection[2]*100., 2))
                    cv2.putText(frame[1], label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                tracker_bbox = self.tracker.bbox
                if tracker_bbox is not None:
                    (left, top, right, bottom) = tracker_bbox
                    cv2.rectangle(frame[1], (int(left), int(top)), (int(right), int(bottom)), (255,255,255), 5)

                cv2.imshow("Video", frame[1])
                if cv2.waitKey(1) == ord("q"):
                    self.stopped = True

    def stop(self):
        self.stopped = True        
