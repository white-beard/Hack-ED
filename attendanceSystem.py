import cv2
import numpy as np
import face_recognition
import json
import os
import time
import pandas as pd
from collections import Counter

with open("config.json", "r") as file:
    params = json.load(file)["Params"]

# I/p : path to the student face images folder
# O/p : list of images, names
def collect_names_and_images(students_images):
    images = []; names = []
    for img in os.listdir(students_images):
        curImage = cv2.imread(f"{students_images}/{img}")
        images.append(curImage)
        names.append(os.path.splitext(img)[0])
    return images, names

# I/p : list of images which is created in collect_names_and_images function
# O/p : list of image encoding of the images inputted
def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

def create_attendance_sheet(students):
    df = {"Names": students}
    df = pd.DataFrame(df)
    df.to_csv(params["attendanceSheet"] + "/" + "attendanceSheet.csv", index = False)

class ClassRoomAttendance:

    def __init__(self):
        self.video = os.path.join(
            params["upload_folder"], os.listdir(params["upload_folder"])[0])
        self.video = cv2.VideoCapture(self.video)       
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_folder = params["images_folder"]
        self.video_folder = params["video_folder"]
        self.fps = 1
        self.video_name = "Evidence.mp4"
        self.count = 0
        self.list_images, self.StudentNames = collect_names_and_images(params["studentImages"])
        self.listofknownEncodes = findEncodings(self.list_images)
        self.students = []

    def __del__(self):
        self.video.release()

    def images(self):
        while True:
            ret, img = self.video.read()
            
            if ret:
                # print(ret)
                # print("Analyzing Video and collecting frames")
                imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(self.listofknownEncodes, encodeFace)
                    faceDis = face_recognition.face_distance(self.listofknownEncodes, encodeFace)

                    # It will give 
                    matchindex = np.argmin(faceDis)

                    if matches[matchindex]:
                        name = self.StudentNames[matchindex].upper()
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        
                    else:    
                        name = "Unindentified"
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)

                    cv2.putText(img, name, (x1+6, y2-6), self.font, 1, (255, 255, 255), 2)
                    self.count += 1
                    if len(name)>0:                       
                        self.students.append(name)
                    print(f"Students : {self.students}")
                    cv2.imwrite(params["images_folder"]+"/" +
                            f"{self.count}.jpg", img)
                            

            else:
                print(f"students in else : {self.students}")
                self.students = list(set(self.students))
                print(f"Students list = {self.students}")
                create_attendance_sheet(self.students)
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
            height, width, _ = img.shape
            size = (width, height)
            for k in range(2):
                frame_arrays.append(img)

        out = cv2.VideoWriter(
            params["video_folder"]+"/"+self.video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        for i in range(len(frame_arrays)):
            out.write(frame_arrays[i])
        out.release()
