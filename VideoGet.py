from threading import Thread
import cv2
import prctl
import time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.src = src
        self.stopped = False
        self.frame = None

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        print("XXXXXXXXXXXXXXXXXXXXXXXXX")
        prctl.set_name('VideoGet')
        self.stream = cv2.VideoCapture(self.src)
        self.frameno = 0
        (self.grabbed, frame) = self.stream.read()
        self.frame = (self.frameno, frame)
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                self.frame = (self.frameno, frame)
            time.sleep(0.1)
                
    def stop(self):
        self.stopped = True        
