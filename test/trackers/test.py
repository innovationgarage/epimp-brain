import cv2
import numpy as np
import skvideo.io 
import sys
import time

kws = {
    "tracker": "MIL",
    "scale": 1}
args = []
for arg in sys.argv[1:]:
    if arg.startswith("--"):
        arg = arg[2:]
        value = True
        if "=" in arg:
            arg, value = arg.split("=", 1)
        kws[arg] = value
    else:
        args.append(arg)

if kws.get("help", False):
    print("""Options:
--tracker=%(trackers)s
--scale=1
--gray
""" % {"trackers": "|".join(name[len("Tracker"):-len("_create")] for name in dir(cv2) if "Tracker" in name and "create" in name)})
    sys.exit(1)
        
tracker = None

cap = skvideo.io.vreader("cocecan.webm")

if not cap:
  print("Error opening video stream or file")

# xmin,ymin,w,h
bbox = 712, 170, 118, 239

factor = int(kws["scale"])
gray = not not kws.get("gray", False)
trackercls = kws["tracker"]

for idx, frame in enumerate(cap):
    if idx > 100:
        print("Speed %s/%s = %s frames/s" % (idx, t1-t0, idx / (t1-t0)))
        break
    trackerframe = frame
    if gray:
        trackerframe = cv2.cvtColor(trackerframe, cv2.COLOR_BGR2GRAY)
    trackerframe = cv2.resize(trackerframe, (int(trackerframe.shape[1] / factor), int(trackerframe.shape[0] / factor)))
    #print(trackerframe.shape)
    
    if not tracker:

        print("Tracker=%s, factor=%s, gray=%s, shape=%s" % (trackercls, factor, gray, trackerframe.shape))
        
        tracker = getattr(cv2, "Tracker%s_create" % trackercls)()
        trackerbbox = [int(c/factor) for c in bbox]
        ok = tracker.init(trackerframe, tuple(trackerbbox))
        t0 = time.time()
    else:
        ok, bbox = tracker.update(trackerframe)
        t1 = time.time()
#        if idx % 10 == 0:
#            print("Speed %s/%s = %s frames/s" % (idx, t1-t0, idx / (t1-t0)))
        
    x, y, w, h = [int(c*factor) for c in bbox]
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sys.exit(1)
