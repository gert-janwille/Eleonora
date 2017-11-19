"""
This script will preprocess the facial images choose the faces front side
and move them to a new folder and renaming to the expression.

using the KDEF Dataset (http://www.emotionlab.se/resources/kdef)

    Example: AF01ANS.JPG Letter 1: Session

    This script looks to the file name.
    The last character is the position
         S => Front side
    The 2 characters before are the Emotion.
"""

import os

afraidCounter = 0
angryCounter = 0
disgustedCounter = 0
happyCounter = 0
neutralCounter = 0
sadCounter = 0
surprisedCounter = 0

if __name__ == '__main__':
    rootdir = input("Drag & Drop INPUT folder > ")
    rootdir = rootdir.strip() + "/"

    outPath = input("Drag & Drop OUTPUT folder > ")
    outPath = outPath.strip() + "/"

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filePath = os.path.join(subdir, file)

            if file.endswith('AFS.JPG'):
                os.rename(filePath, outPath + "afraid_" + str(afraidCounter) + ".jpg")
                afraidCounter = afraidCounter + 1

            if file.endswith('ANS.JPG'):
                os.rename(filePath, outPath + "angry_" + str(angryCounter) + ".jpg")
                angryCounter = angryCounter + 1

            if file.endswith('DIS.JPG'):
                os.rename(filePath, outPath + "disgusted_" + str(disgustedCounter) + ".jpg")
                disgustedCounter = disgustedCounter + 1

            if file.endswith('HAS.JPG'):
                os.rename(filePath, outPath + "happy_" + str(happyCounter) + ".jpg")
                happyCounter = happyCounter + 1

            if file.endswith('NES.JPG'):
                os.rename(filePath, outPath + "neutral_" + str(neutralCounter) + ".jpg")
                neutralCounter = neutralCounter + 1

            if file.endswith('SAS.JPG'):
                os.rename(filePath, outPath + "sad_" + str(sadCounter) + ".jpg")
                sadCounter = sadCounter + 1

            if file.endswith('SUS.JPG'):
                os.rename(filePath, outPath + "surprised_" + str(surprisedCounter) + ".jpg")
                surprisedCounter = surprisedCounter + 1

    print("afraid:", afraidCounter)
    print("angry:", angryCounter)
    print("disgusted:", disgustedCounter)
    print("happy:", happyCounter)
    print("neutral:", neutralCounter)
    print("sad:", sadCounter)
    print("surprised:", surprisedCounter)

    total = afraidCounter + angryCounter + disgustedCounter + happyCounter + neutralCounter + sadCounter + surprisedCounter
    print("\n Total", total)
