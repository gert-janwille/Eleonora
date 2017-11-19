"""
This script will preprocess the facial images to train
a neural network.
"""

import cv2
import os

FACE_CASCADE_FILE = os.path.join('../eleonora/data/haarcascades/','haarcascade_frontalface_alt.xml')

def facechop(image, outPath):
    ar = image.split('/')
    fileName = ar[len(ar)-1].split('.')[0]

    cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)

    img = cv2.imread(image)

    frame = cv2.flip(img, 1)

    minisize = (frame.shape[1],frame.shape[0])
    miniframe = cv2.resize(frame, minisize)
    gray = cv2.cvtColor(miniframe, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(gray, (x,y), (x+w,y+h), (255,255,255))

        print("Face Found:", fileName, "\n")
        sub_face = gray[y:y+h, x:x+w]
        face_file_name = outPath + fileName + ".jpg"
        os.remove(image)
        cv2.imwrite(face_file_name, sub_face)


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


    print("Found Images:", imageCounter)
