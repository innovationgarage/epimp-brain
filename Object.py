from datetime import datetime
import time
from math import *
import numpy as np
import math_tools as mtools

class D2(object):
    "D2 laptop:86 -2424 -1024 86 -1024 86 186 86 -1024"
    "D2 id x1 y1 x2 y2 x3 y3 x4 y4 extra"
    def __init__(self, line):
        self.timestamp = time.time() #sec
        self._type = line[1].split(':')[0]
        self.prob = float(line[1].split(':')[1])
        self.x_1 = float(line[3])
        self.y_1 = float(line[4])
        self.x_2 = float(line[5])
        self.y_2 = float(line[6])
        self.x_3 = float(line[7])
        self.y_3 = float(line[8])
        self.x_4 = float(line[8])
        self.y_4 = float(line[9])
        self.properties = {
            '_type': self._type,
            'x_1': self.x_1,
            'y_1': self.y_1,
            'x_2': self.x_2,
            'y_2': self.y_2,
            'x_3': self.x_3,
            'y_3': self.y_3,
            'x_4': self.x_4,
            'y_4': self.y_4,
            'prob': self.prob,
            'timestamp': self.timestamp
        }
                            
    def getProperties(self):
        props = {
            'x_1': self.x_1,
            'y_1': self.y_1,
            'x_2': self.x_2,
            'y_2': self.y_2,
            'x_3': self.x_3,
            'y_3': self.y_3,
            'x_4': self.x_4,
            'y_4': self.y_4,
            'prob': self.prob,
            'timestamp': self.timestamp
        }
        return props

class T2(object):
    "T2 600 350"
    "T2 x y"
    def __init__(self, line):
        self.timestamp = time.time() #sec
        self.x = float(line[1])
        self.y = float(line[2])
        self.properties = {
            'x': self.x,
            'y': self.y,
            'timestamp': self.timestamp
        }
                            
    def getProperties(self):
        props = {
            'x': self.x,
            'y': self.y,
            'timestamp': self.timestamp
        }
        return props

        
        
