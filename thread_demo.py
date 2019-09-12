import argparse
import cv2
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow
from ObjectDetect import ObjectDetect
import time

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def threadDetect(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    modelname = 'TFModels/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    configname = 'TFModels/ssd_mobilenet_v1_ppn_coco.pbtxt'
    classnames = 'TFModels/coconew.names'
    garbageclasses = [
        "shoe", "hat", "eye glasses", "frisbee",
        "bottle", "plate", "wine glass", "cup", "fork", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "fruit",
        "hotdog", "pizza", "donut", "cake",
        "vase", "scissors", "toothbrush", "cardboard", "napkin",
        "net", "paper", "plastic", "straw"
    ]    
    with open(classnames, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    cvNet = cv2.dnn.readNetFromTensorflow(modelname, configname)
    threshold = 0.5

    video_getter = VideoGet(source).start()
    object_detector = ObjectDetect(cvNet, threshold, classes, garbageclasses, video_getter).start()
    video_shower = VideoShow(video_getter, object_detector).start()
    
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or object_detector.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            object_detector.stop()
            break

        time.sleep(1)
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default=0,
        help="Path to video file or integer representing webcam index"
            + " (default 0).")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
            + " show (video show in its own thread), both"
            + " (video read and video show in their own threads),"
            + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    if args["thread"] == "both":
        threadBoth(args["source"])
    elif args["thread"] == "get":
        threadVideoGet(args["source"])
    elif args["thread"] == "show":
        threadVideoShow(args["source"])
    elif args["thread"] == "detect":
        threadDetect(args["source"])
    else:
        noThreading(args["source"])

if __name__ == "__main__":
    main()
