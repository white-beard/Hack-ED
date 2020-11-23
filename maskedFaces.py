from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import cv2
import numpy as np
import os

import json
import imutils


with open("config.json", 'r') as file:
    params = json.load(file)["Params"]


faceNet = cv2.dnn.readNet(
    params["facenet_prototxt"], params["facenet_weights"])
maskNet = load_model(params["maskedVsUnmasked_weights"])


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []; locs = []; preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")

            (startx, starty) = (max(0, startx), max(0, starty))
            (endx, endy) = (min(w-1, endx), min(h-1, endy))


            face = frame[starty:endy, startx:endx]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startx, starty, endx, endy))

    if len(faces)>0:
        faces = np.array(faces, dtype = "float32")
        preds = maskNet.predict(faces, batch_size = 32)
    return (locs, preds)


	


class MaskedFacesClassifier:

    def __init__(self):
        self.video = os.path.join(params["upload_folder"], os.listdir(params["upload_folder"])[0])
        self.video = cv2.VideoCapture(self.video)
        # self.predictor = dlib.shape_predictor(params["dlib_68_face_landmarks"])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_folder = params["images_folder"]
        self.video_folder = params["video_folder"]
        self.fps = 1
        self.video_name = "Evidence.mp4"
        self.count = 0      
        

    def __del__(self):
        self.video.release()

    def images(self):
        while True:
            ret, frame = self.video.read()

            if ret:    
               frame = imutils.resize(frame, width=400)
               (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
               for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (withoutmask,mask) = pred
                    label = "Mask" if mask>withoutmask else "No Mask"
                    color = (0, 255, 0) if label=="Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(withoutmask,mask) * 100)
                    cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    self.count += 1
                    cv2.imwrite(params["images_folder"]+"/"+f"{self.count}.jpg", frame)
                    
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

