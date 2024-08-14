from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#Live capture of video
# cap=cv2.VideoCapture(0) # id no of your webcam,if only 1 webcam ;0

# cap.set(3,640) #3 for width
# cap.set(4,480) #4 for height

#Saved videos
cap=cv2.VideoCapture("C:\\Users\\Admin\\Desktop\\ObjectDetection\\Videos\\cars.mp4")

model=YOLO("../Yolo-Weights/yolov8l.pt")

className=['person','bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck','boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball','kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass','cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',  'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
             'bed','dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator','book', 'clock', 'vase', 'scissors','teddy bear',
               'hair drier', 'toothbrush']

mask=cv2.imread("C:\\Users\\Admin\\Desktop\\ObjectDetection\\CarCounter\q\mask1.png")

#Tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
while True:
    success,img=cap.read() #reads a frame
    # img = cv2.imread("C://Users//Admin//Desktop//ObjectDetection//Images//2.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Masked Image: ", mask)
    # print("Mask",mask.shape[:2])
    print("Img",img.shape[:2])
    mask_resize=cv2.resize(mask,(1280,720)) 

    imgRegion=cv2.bitwise_and(img,mask_resize)
    # mask_resize=cv2.resize(mask,(640,480))

    results=model(img,stream=True)
    # print("Results : ", results)
    detections=np.empty((0,5))
    for r in results:
        # print("Resuls R: ", r)
        boxes=r.boxes
        # print("Boxes: ", boxes)
        for box in boxes:
            # x1,y1,w,h=box.xywh[0]
            # bbox=int(x1),int(y1),int(w),int(h)
           
            #Bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print("Cordinates: ", (x1, y1), (x2, y2))
            
            w,h=x2-x1,y2-y1

            cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            print(conf)

            #CLass Name
            cls=int(box.cls[0])
            currentClass=className[cls]

            if currentClass=="car" or currentClass=='truck' or currentClass=='bus' or currentClass=='motorbike' and conf>0.3:
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(0,y1)),scale=1,thickness=1,offset=5) 
                 #offset=5: Adds a 5-pixel margin around the text inside the rectangle.
                cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h=x2-x1,y2-y1 
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f'{id}',(max(0,x1),max(0,y1)),scale=1,thickness=1,offset=5) 

    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion",imgRegion)

   # cv2.waitKey(1) # 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


