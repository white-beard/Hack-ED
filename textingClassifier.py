import cv2
import numpy as np
import imutils
import os
import json
import random
from Models.textingClassfierModel import Texting_VS_Normal

with open("config.json", 'r') as file:
    params = json.load(file)["Params"]


class TextingClassifier:

    def __init__(self):
        self.video = os.path.join(
            params["upload_folder"], os.listdir(params["upload_folder"])[0])
        self.video = cv2.VideoCapture(self.video)
        self.textingClassifier = Texting_VS_Normal(params["textingClassifier_arch"], params["textingClassifier_weights"])
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
                self.count += 1
                img = cv2.resize(frame, (100, 100))
                print(f"Image Shape : {img.shape}")
                print("\n")             
                prediction, preds_proba = self.textingClassifier.predict(img)
                print(f"Prediction : {prediction} and probability : {preds_proba[0][0]*100}")
                cv2.putText(frame, f"Status is {prediction} with probablity {preds_proba[0][0]*100}%" , (5, 30), self.font, 1.2, (0, 220, 13), 2)
                cv2.imwrite(params["images_folder"]+"/" +
                            f"{self.count}.jpg", frame)

            else:
                break

    def generate_video(self):
        print("Generating Video")
        frame_arrays = []
        files = [f for f in os.listdir(params["images_folder"]) if os.path.isfile(
            os.path.join(params["images_folder"], f))]
        files.sort(key=lambda x: int(x.split(".")[0]))
        for i in range(len(files)):
            filename = params["images_folder"] + "/" + files[i]
            """reading Images"""
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            for k in range(2):
                frame_arrays.append(img)

        out = cv2.VideoWriter(
            params["video_folder"]+"/"+self.video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        for i in range(len(frame_arrays)):
            out.write(frame_arrays[i])
        out.release()
