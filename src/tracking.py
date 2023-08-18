import datetime
from ultralytics import YOLO
import cv2
import pyvirtualcam
import numpy as np
from KalmanFilter import KalmanFilter

def bounds(x, y, relative_w, relative_h, real_w, real_h):
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    if x < relative_w:
        xmin = 0
        xmax = relative_w*2
    elif x > (real_w-relative_w):
        xmin = real_w-(relative_w*2)
        xmax = real_w
    else:
        xmin = int(x-relative_w)
        xmax = int(x+relative_w)
    
    if y < relative_h:
        ymin = 0
        ymax = relative_h*2
    elif y > (real_h-relative_h):
        ymin = real_h-(relative_h*2)
        ymax = real_h
    else:
        ymin = int(y-relative_h)
        ymax = int(y+relative_h)
    return xmin, xmax, ymin, ymax



# initialize the video capture object
#video_cap = cv2.VideoCapture("Videos/test1.mp4")

video_cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_cap.isOpened():
    raise IOError("Cannot open webcam")

# Video from BGR to RGB
fmt = pyvirtualcam.PixelFormat.BGR

# load the pre-trained YOLOv8n model
#model = YOLO("yolov8n.pt")
model = YOLO("yolov8n-pose.pt")
model.to('cuda')

# Kalman filter
KF = KalmanFilter(0.1, 1, 1, 5, 20, 20)
# define some constants
CONFIDENCE_THRESHOLD = 0.6
w_fhd = int(1920/2)
h_fhd = int(1080/2)
w_hd=int(1280/2)
h_hd=int(720/2)
debugging = True
centroidX = 0
centroidY = 0
keyPoints = [0, 5, 6]
(x1, y1) = KF.update([w_fhd, h_fhd])


with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=fmt) as cam:
    while True:
        # Fotograma vacio
        frameProcessed = np.zeros((cam.height, cam.width, 3), np.uint8)
        
        # Inicio de tiempo para medir
        start = datetime.datetime.now()
    
        # Captura de video
        ret, frame = video_cap.read()

        # Si no hay más frames, se cierra el bucle
        if not ret:
            break
        
        # Se corre YOLO
        detections = model(frame, classes=0, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].keypoints.data.tolist()
        
        temp_centroidX = 0
        temp_centroidY = 0
        distanceCenter = 0
        
        for data in detections:
            for i in keyPoints:
                temp_centroidX = temp_centroidX + data[i][0]
                temp_centroidY = temp_centroidY + data[i][1]
                cv2.circle(frame, (int(data[i][0]), int(data[i][1])), radius=5, color=(255, 0, 0), thickness=-1)
            temp_centroidX = temp_centroidX/3
            temp_centroidY = temp_centroidY/3
            temp_distanceCenter = np.sqrt((temp_centroidX-w_fhd)**2+(temp_centroidY-h_fhd)**2)
            if distanceCenter == 0:
                distanceCenter = temp_distanceCenter
                centroidX = int(temp_centroidX)
                centroidY = int(temp_centroidY)
            elif temp_distanceCenter < distanceCenter:
                distanceCenter = temp_distanceCenter
                centroidX = int(temp_centroidX)
                centroidY = int(temp_centroidY)
                
        focus = np.asarray(KF.predict()[0]).reshape(-1)

        if debugging:
            cv2.circle(frame, (centroidX, centroidY), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (int(focus[0]), int(focus[1])), radius=5, color=(0, 255, 0), thickness=-1)
        
        (x1, y1) = KF.update([centroidX, centroidY])
        # Calculo del enfoque 
        xmin_p, xmax_p, ymin_p, ymax_p = bounds(int(focus[0]), int(focus[1]), w_hd, h_hd, 1920, 1080)
        frame = frame[ymin_p:ymax_p, xmin_p:xmax_p]

        # Aumentado del tamaño
        frame = cv2.resize(frame, (1920, 1080))
        
        if debugging:           
            # end time to compute the fps
            end = datetime.datetime.now()
            # show the time it took to process 1 frame
            total = (end - start).total_seconds()
            # calculate the frame per second and draw it on the frame
            fps = f"FPS: {1 / total:.2f}"
            cv2.putText(frame, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        cam.send(frame)
        cam.sleep_until_next_frame()

    video_cap.release()