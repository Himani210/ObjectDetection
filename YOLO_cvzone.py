from ultralytics import YOLO
import cv2
import cvzone
import math

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

while True:
    success,img=cap.read() #reads a frame
    # img = cv2.imread("C://Users//Admin//Desktop//ObjectDetection//Images//2.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=model(img,stream=True)
    print("Results : ", results)
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

            cvzone.cornerRect(img,(x1,y1,w,h))

            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            print(conf)

            #CLass Name
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{className[cls]} {conf}',(max(0,x1),max(0,y1)),scale=1,thickness=1)

    cv2.imshow("Image",img)
   # cv2.waitKey(1) # 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


