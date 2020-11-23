from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import json
import os

with open("config.json", 'r') as file:
    params = json.load(file)["Params"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B)/(2*C)
    return ear




class DrowsyDetector:

    def __init__(self):
        self.video = os.path.join(params["upload_folder"], os.listdir(params["upload_folder"])[0])
        self.video = cv2.VideoCapture(self.video)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(params["dlib_68_face_landmarks"])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_folder = params["images_folder"]
        self.video_folder = params["video_folder"]
        self.fps = 1
        self.video_name = "Evidence.mp4"
        self.count = 0
        self.EYE_AR_THRES = 0.3
        self.EYE_AR_CONSEC_FRAME = 48
        self.COUNTER = 0
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        

    def __del__(self):
        self.video.release()

    def images(self):
        while True:
            ret, frame = self.video.read()

            if ret:                
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]

                    leftEar = eye_aspect_ratio(leftEye)
                    rightEar = eye_aspect_ratio(rightEye)

                    ear = 0.5*(leftEar + rightEar)

                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)

                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 200, 124), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 200, 124), 1)
                    
                    

                    if ear < self.EYE_AR_THRES:
                        self.COUNTER += 1
                        self.count += 1

                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAME:
                            self.count += 1                
                            cv2.putText(frame, "DROWSY ALERT", (10, 30), self.font, 0.5, (255, 0, 0), 2)
                            cv2.imwrite(params['images_folder']+"/"+f"{self.count}.jpg", frame)

                    else:
                        self.count += 1
                        self.COUNTER = 0
                    
                    cv2.putText(frame, f"EAR: {round(ear, 2)}", (300, 30),self.font, 0.7, (200, 0, 0), 3)
                    cv2.imwrite(params['images_folder']+"/"+f"{self.count}.jpg", frame)                

            else:
                break    
    
    def generate_video(self):
        print("Generating Video")
        frame_arrays = []
        files = [f for f in os.listdir(params["images_folder"]) if os.path.isfile(os.path.join(params["images_folder"], f))]
        files.sort(key=lambda x: int(x.split(".")[0]))
        for i in range(len(files)):
            filename = params["images_folder"] + "/" +files[i]
            """reading Images"""
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            for k in range(2):
                frame_arrays.append(img)

        out = cv2.VideoWriter(params["video_folder"]+"/"+self.video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        for i in range(len(frame_arrays)):           
            out.write(frame_arrays[i])        
        out.release()

