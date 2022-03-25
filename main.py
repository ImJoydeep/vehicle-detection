import cv2 as cv
#OpenCV DNN
net = cv.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
#Load Class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#Initialize Camera
cap = cv.VideoCapture("flyover.mp4")
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#FULL HD 1920 x 1080
writer = cv.VideoWriter('vehicle_detection_DEMO.mp4',cv.VideoWriter_fourcc(*'DVIX'),20,(width,height))
while True:
    #Get Frames
    ret, frame = cap.read()
    #writer.write(frame)

    #Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, scores, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        if class_name in ["car","motorbike","bus","bicycle"]:
            cv.putText(frame, class_name,(x, y-10), cv.FILE_NODE_REAL, 2, (255, 0, 50), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 50), 3)

    cv.imshow('Frame', frame)
    writer.write(frame)
    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
writer.release()
cv.destroyAllWindows()
