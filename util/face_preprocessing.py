"""
This script will preprocess the facial images to train
a neural network.
"""

<<<<<<< HEAD
import os
import cv2
import numpy as np

FACE_CASCADE_FILE = os.path.join('../eleonora/data/haarcascades/','haarcascade_frontalface_alt.xml')
foundFace = 0

def facechop(image, outPath):
    global foundFace

=======
import cv2
import os

FACE_CASCADE_FILE = os.path.join('../eleonora/data/haarcascades/','haarcascade_frontalface_alt.xml')

def facechop(image, outPath):
>>>>>>> dev
    ar = image.split('/')
    fileName = ar[len(ar)-1].split('.')[0]

    cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)

    img = cv2.imread(image)

    frame = cv2.flip(img, 1)
<<<<<<< HEAD
    minisize = (200, 200)
    img_scaled = cv2.resize(frame, minisize, fx=0.25, fy=0.25, interpolation = cv2.INTER_LINEAR)


    # miniframe = cv2.resize(img_scaled, minisize)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
=======

    minisize = (frame.shape[1],frame.shape[0])
    miniframe = cv2.resize(frame, minisize)
    gray = cv2.cvtColor(miniframe, cv2.COLOR_BGR2GRAY)
>>>>>>> dev

    faces = cascade.detectMultiScale(gray)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(gray, (x,y), (x+w,y+h), (255,255,255))

        print("Face Found:", fileName, "\n")
        sub_face = gray[y:y+h, x:x+w]
<<<<<<< HEAD
        face_file_name = outPath + "_processed/" + fileName + ".jpg"
        cv2.imwrite(face_file_name, sub_face)
        foundFace = foundFace + 1
=======
        face_file_name = outPath + fileName + ".jpg"
        os.remove(image)
        cv2.imwrite(face_file_name, sub_face)
>>>>>>> dev


if __name__ == '__main__':
    path = input("Drag & Drop folder > ")
    path = path.strip() + "/"

    valid_images = [".jpg",".gif",".png"]
    imageCounter = 0;

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]

        if ext.lower() not in valid_images:
            continue

        imageCounter = imageCounter + 1
        print("Image:", imageCounter, " path:", os.path.join(path,f))
        facechop(os.path.join(path,f), path)


<<<<<<< HEAD
    print("Found Faces: %s/%s " % (foundFace, imageCounter))
=======
    print("Found Images:", imageCounter)
>>>>>>> dev
