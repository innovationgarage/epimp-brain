import cv2 as cv
import time
from coco_classes import classes

collect_classes = ["bottle", "frisbee"]
avoid_classes = ["person", "bird", "cat", "dog", "backpack", "umbrella", "handbag", "cell phone"]

# frozen_weights = "TFModels/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb"
# model_config = "TFModels/ssd_mobilenet_v1_ppn_coco.pbtxt"

frozen_weights = "TFModels/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
model_config = "TFModels/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"

cvNet = cv.dnn.readNetFromTensorflow(frozen_weights, model_config)

def detect(img):
    t0 = time.time()
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            category = classes[str(int(detection[1]))]
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv.putText(img, category, (int(left),int(top)), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        
        # t1 = time.time()
        # print('dt = {} seconds'.format(t1-t0))

img = cv.imread('bottle2.jpg')
# for filename in ["example.jpg", "beach1.jpg", "beach2.jpg"]:
#     img = cv.imread(filename)
detect(img)
cv.imshow('img', img)
cv.waitKey()

# vs = cv.VideoCapture(0)
# while True:
#     _, img = vs.read()
#     if img is None:
#         break

