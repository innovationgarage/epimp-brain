from threading import Thread
import cv2
import prctl
import time
import collections

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.src = src
        self.stopped = False
        self.frame = None
        self.frames = []

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        prctl.set_name('VideoGet')
        self.stream = cv2.VideoCapture(self.src)
        self.frameno = 0
        (self.grabbed, frame) = self.stream.read()
        assert frame is not None, "Unable to open Vide stream"
        self.frame = (self.frameno, frame)
        print('W, H', self.frame[1].shape)
        self.frames.append(self.frame)
        self.cnt = 0
        while not self.stopped:
            self.cnt += 1
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                self.frame = (self.frameno, frame)
                self.frames.append(self.frame)
            time.sleep(0.1)

    def record(self):
        res = self.frames
        assert self.frame is not None, "Frames not initialized at record at frame no %s" % self.cnt
        self.frames = [self.frame]
        return res
        
    def stop(self):
        self.stopped = True        
