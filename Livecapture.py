from ultralytics import YOLO
import cv2
import cvzone

cap=cv2.VideoCapture(0) # id no of your webcam,if only 1 webcam ;0

cap.set(3,640) #3 for width
cap.set(4,480) #4 for height

model=YOLO("../Yolo-Weights/yolov8l.pt")

while True:
    success,img=cap.read()
    results=model(img,stream=True) #Passes the captured frame to the YOLO model for object detection. 
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
    cv2.imshow("Image",img)
   # cv2.waitKey(1) # 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 