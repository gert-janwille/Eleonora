import os
import random

from skimage import transform
from skimage.io import imread, imshow
from eleonora.common.constants import *

dataset = []

def loadDataset(path, img_rows, img_cols, SPLIT_PERCENT):
    # import images and label, put them in an dict
    valid_images = [".jpg",".gif",".png"]
    c = 1

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        resized = transform.resize(imread(os.path.join(path, f)),(img_rows, img_cols), mode='constant')
        dataset.append({"img": resized, "label": f.split("_")[0]})

        wait = int(0 + (100 - 0) * (c - 0) / ((len(os.listdir(path))-1) - 0))
        print(" Initializing Dataset: %s%s%s%s" % (T, wait, "%", W), end="\r")
        c = c + 1

    # Shuffle the array
    random.shuffle(dataset)

    # Get the featues and target
    features, target = getFeatuesAndTargets(dataset)

    if len(target) != len(features):
        print(B + "Something is wrong with the Dataset\n" + W)
        sys.exit()

    # Split into training and test set
    training_images, test_images, y_train, y_test = splitDataset(features, target, SPLIT_PERCENT)


    print("\tTraining Set:", len(training_images), len(y_train))
    print("\tTest Set:", len(test_images), len(y_test))

    return training_images, test_images, y_train, y_test

def getFeatuesAndTargets(dataset):
    features, target = [], []

    for i in range(len(dataset)):
        features.append(dataset[i]["img"])

        if dataset[i]["label"] == "afraid":
            target.append([1, 0, 0, 0, 0, 0, 0])
        if dataset[i]["label"] == "angry":
            target.append([0, 1, 0, 0, 0, 0, 0])
        if dataset[i]["label"] == "disgusted":
            target.append([0, 0, 1, 0, 0, 0, 0])
        if dataset[i]["label"] == "happy":
            target.append([0, 0, 0, 1, 0, 0, 0])
        if dataset[i]["label"] == "neutral":
            target.append([0, 0, 0, 0, 1, 0, 0])
        if dataset[i]["label"] == "sad":
            target.append([0, 0, 0, 0, 0, 1, 0])
        if dataset[i]["label"] == "surprised":
            target.append([0, 0, 0, 0, 0, 0, 1])

    # Print the length of targets en features
    print(T, "\nShape:", len(target), len(features), W)

    return features, target

def splitDataset(features, target, percent):
    split_number = int((len(target) / 100) * percent)
    training_images, test_images, y_train, y_test = features[:split_number], features[split_number:], target[:split_number], target[split_number:]
    return training_images, test_images, y_train, y_test
