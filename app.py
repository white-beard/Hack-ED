from flask import Flask, request, render_template, redirect, Response, send_from_directory, abort
from werkzeug.utils import secure_filename

from ped_detect import StudentOutsideDorms
from drowsyDetector import DrowsyDetector
from gender import GenderClassifier
from textingClassifier import TextingClassifier
from drinkingClassifier import DrinkingClassifier
from maskedFaces import MaskedFacesClassifier
from attendanceSystem import ClassRoomAttendance


import os
import json
import glob

from zipfile import ZipFile

app = Flask(__name__)

with open("config.json", 'r') as file:
    params = json.load(file)["Params"]


app.config["MAX_VIDEO_LENGTH"] = int(params["max_video_length"])  # user can upload upto 100MB
app.config["UPLOAD_FOLDER"] = params["upload_folder"]
app.config["ALLOWED_EXTENSION"] = params["allowed_extension"]
app.config["ZIP_EXTENSION"] = params["zip_extensions"]
app.config["ZIP_FOLDER"] = params["zip_upload"]

@app.route("/home")
@app.route("/")
def index():
    return render_template("index.html", home="active", steps="", about="", getStarted="")


@app.route("/getStarted")
def getStarted():
        try:
            last_uploads = os.listdir(app.config["UPLOAD_FOLDER"])
            if len(last_uploads)>0:
                for file in last_uploads:      
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], file))


            last_images = os.listdir(params["images_folder"])
            if len(last_images)>0:
                for file in last_images:      
                    os.remove(os.path.join(params["images_folder"], file))
            
            last_video = os.listdir(params["video_folder"])
            if len(last_video)>0:
                for file in last_video:      
                    os.remove(os.path.join(params["video_folder"], file))

            last_zipuploads = os.listdir(params["zip_upload"])
            if len(last_zipuploads)>0:
                for file in last_zipuploads:      
                    os.remove(os.path.join(params["zip_upload"], file))
            
            last_studentimages = os.listdir(params["studentImages"])
            if len(last_studentimages)>0:
                for file in last_studentimages:      
                    os.remove(os.path.join(params["studentImages"], file))

            last_attendanceSheet = os.listdir(params["attendanceSheet"])
            if len(last_attendanceSheet)>0:
                for file in last_attendanceSheet:      
                    os.remove(os.path.join(params["attendanceSheet"], file))


            return render_template("getStarted.html", home="", steps="", about="", getStarted="active")
        except TypeError:
            return render_template("getStarted.html", home="", steps="", about="", getStarted="active")


@app.route("/steps")
def steps():
    return render_template("steps.html", home="", steps="active", about="", getStarted="")


@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    try:
        last_uploads = os.listdir(app.config["UPLOAD_FOLDER"])
        if len(last_uploads)>0:
            for file in last_uploads:      
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], file))

        last_images = os.listdir(params["images_folder"])
        if len(last_images)>0:
            for file in last_images:      
                os.remove(os.path.join(params["images_folder"], file))
                print("images_folder is clean")
        
        last_video = os.listdir(params["video_folder"])
        if len(last_video)>0:
            for file in last_video:      
                os.remove(os.path.join(params["video_folder"], file))
                print("video folder is clean")

        if (request.method == "GET"):
            return redirect("/getStarted")


        if (request.method == "POST"):
            file = request.files['video']
            option = request.form.get("option")
            print(f"User uploaded:\n==========\n{file.filename}\n==========")
            filename = secure_filename(file.filename)

            if filename == "":
                return redirect("/uploader")

            if (filename.split(".")[-1]).upper() not in app.config["ALLOWED_EXTENSION"]:
                return redirect("/uploader")

            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            print(option)
            if (option == "Students Outisde Dorms"):
                print("redirecting to studentoutsideDorms")
                return redirect("/studentOutsideDorms")    

            if (option == "Drowsy Driver"):
                print("redirecting to Drowsydetection")
                return redirect("/drowsyDetection")

            if (option == "Texting while driving"):
                print("redirecting to texting")
                return redirect("/textingWhileDriving")

            if (option == "Drinking while driving"):
                print("redirecting to drinking")
                return redirect("/drinkingWhileDriving")

            if (option == "Gender Classification"):
                print("redirecting to genderClassification")
                return redirect("/genderClassification")
            
            if (option == "Masked faces Classification"):
                print("redirecting to maskedFaceClasification")
                return redirect("/maskedFaceClasification")  

            if (option == "ClassRoom Attendance"):
                print("redirecting to classroomattendance")
                return redirect("/classroomattendance")

    except TypeError:
        return redirect("/getStarted")


@app.route("/classroomattendance")
def classroomattendanceSystem():
    return render_template("classroomattendance.html")

@app.route("/attendanceImages", methods=["GET", "POST"])
def saveAttendanceImagesUploads():

    try:
        
        if (request.method == "GET"):
            return redirect("/classroomattendance")

        if (request.method == "POST"):
            file = request.files['zipfile']
        
            print(f"User uploaded :\n==========\n{file.filename}\n==========")
            filename = secure_filename(file.filename)

            if filename == "":
                return redirect("/classroomattendance")

            if (filename.split(".")[-1]).upper() not in app.config["ZIP_EXTENSION"]:
                return redirect("/classroomattendance")

            zip_file_path = os.path.join(app.config["ZIP_FOLDER"], filename)
            file.save(zip_file_path)

            with ZipFile(zip_file_path, 'r') as file:
                file.extractall(params["studentImages"])   
            file.close()
            print("redirecting to attendanceSystem")
            return redirect("/attendanceSystem")              

    except TypeError:
        return redirect("/classroomattendance")

@app.route("/attendanceSystem")
def studentAttendance():
    attendancesystem = ClassRoomAttendance()
    attendancesystem.images()
    attendancesystem.generate_video()
    print("VideoComplete")   
    return render_template("attendanceAnalyzer.html")
    


@app.route("/about")
def about():
    return render_template("about.html", home="", steps="", about="active", getStarted="")


@app.route("/studentOutsideDorms")
def studentOutsideDorms():
    pred_detect = StudentOutsideDorms()
    pred_detect.images()
    pred_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")

@app.route("/drowsyDetection")
def drowsyDetection():
    drowsy_detect = DrowsyDetector()
    drowsy_detect.images()
    drowsy_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")


@app.route("/genderClassification")
def genderClassification():
    gender_detect = GenderClassifier()
    gender_detect.images()
    gender_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")


@app.route("/textingWhileDriving")
def textingVsNormalClassification():
    texting_detect = TextingClassifier()
    texting_detect.images()
    texting_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")

@app.route("/drinkingWhileDriving")
def drinkingVsNormalClassification():
    drinking_detect = DrinkingClassifier()
    drinking_detect.images()
    drinking_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")

@app.route("/maskedFaceClasification")
def maskedVsUnMaskedClassification():
    mask_detect = MaskedFacesClassifier()
    mask_detect.images()
    mask_detect.generate_video() 
    print("VideoComplete")   
    return render_template("analysisComplete.html")





@app.route("/download_video/<filename>")
def download_videofile(filename):
    try:
        return send_from_directory(params["video_folder"], filename = filename, as_attachment=True, cache_timeout = 0)
    except FileNotFoundError:
        abort(404)

@app.route("/download_csv/<filename>")
def download_csvfile(filename):
    try:
        return send_from_directory(params["attendanceSheet"], filename = filename, as_attachment=True, cache_timeout = 0)
    except FileNotFoundError:
        abort(404)




if __name__ == "__main__":
    app.run(debug = True)
