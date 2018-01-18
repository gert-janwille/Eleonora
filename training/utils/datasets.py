import os
import re
import cv2
import random
import numpy as np
import pandas as pd
from scipy import misc
from random import shuffle
from training.utils.constants import *
from keras.utils import np_utils
from collections import defaultdict

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size

        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = './training/data/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = './training/data/fer2013/fer2013.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = './training/data/KDEF/'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    def get_data(self, reduceDatasetNumber = False):
        print("Initializing " + self.dataset_name + " Dataset")
        if self.dataset_name == 'imdb':
            data = self._load_imdb(reduceDatasetNumber)
        elif self.dataset_name == 'fer2013':
            data = self._load_fer2013(reduceDatasetNumber)
        elif self.dataset_name == 'KDEF':
            data = self._load_KDEF()
        return data

    def _load_fer2013(self, reduceDatasetNumber):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        if reduceDatasetNumber != False:
            pixels = pixels[:reduceDatasetNumber]

        width, height = 48, 48
        X = []
        c = 1
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            X.append(face.astype('float32'))
            wait = int(0 + (100 - 0) * (c - 0) / ((len(pixels)-1) - 0))
            print(" Initializing Dataset Faces: %s%s%s%s" % (T, wait, "%", W), end="\r")
            c = c + 1
        X = np.asarray(X)
        X = np.expand_dims(X, -1)
        print("\n")
        y = data['emotion']
        if reduceDatasetNumber != False:
            y = y[:reduceDatasetNumber]
        print(T, "\nShape:", len(X), len(y), W)
        return X, y

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('v.jpg')):
                    print("file Removed", filename)
                    os.remove(os.path.join(folder, filename))
                if filename.lower().endswith(('h.jpg')):
                    print("file Removed", filename)
                    os.remove(os.path.join(folder, filename))
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        c2 = 1
        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = []
        for file_arg, file_path in enumerate(file_paths):
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            try:
                emotion_arg = class_to_arg[file_emotion]
                emotions.append(emotion_arg)
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                faces[file_arg] = image_array
            except:
                continue

            wait = int(0 + (100 - 0) * (c2 - 0) / ((num_faces-1) - 0))
            print(" Initializing Dataset Faces: %s%s%s%s" % (T, wait, "%", W), end="\r")
            c2 = c2 + 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions


Y_SAME = 1
Y_DIFFERENT = 0
IMAGE_CHANNELS = 1

class ImageFile(object):
    def __init__(self, directory, name):
        self.filepath = os.path.join(directory, name)
        self.filename = name
        self.person = filepath_to_person_name(self.filepath)
        self.number = filepath_to_number(self.filepath)

    def get_content(self):
        if IMAGE_CHANNELS == 3:
            img = misc.imread(self.filepath, mode="RGB")
        else:
            img = misc.imread(self.filepath)
            assert len(img.shape) == 2
            img = img[:, :, np.newaxis]
        return img

class ImagePair(object):
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.same_person = (image1.person == image2.person)
        self.same_image = (image1.filepath == image2.filepath)

    def get_key(self, ignore_order):
        fps = [self.image1.filepath, self.image2.filepath]
        if ignore_order:
            key = "$$$".join(sorted(fps))
        else:
            key = "$$$".join(fps)
        return key

    def get_contents(self, height, width):
        img1 = self.image1.get_content()
        img2 = self.image2.get_content()
        if img1.shape[0] != height or img1.shape[1] != width:
            if IMAGE_CHANNELS == 1:
                img1 = misc.imresize(np.squeeze(img1), (height, width))
                img1 = img1[:, :, np.newaxis]
            else:
                img1 = misc.imresize(img1, (height, width))
        if img2.shape[0] != height or img2.shape[1] != width:
            if IMAGE_CHANNELS == 1:
                img2 = misc.imresize(np.squeeze(img2), (height, width))
                img2 = img2[:, :, np.newaxis]
            else:
                img2 = misc.imresize(img2, (height, width))
        return np.array([img1, img2], dtype=np.uint8)

def filepath_to_person_name(filepath):
    last_slash = filepath.rfind("/")
    if last_slash is None:
        return filepath[0:filepath.rfind("_")]
    else:
        return filepath[last_slash+1:filepath.rfind("_")]

def filepath_to_number(filepath):
    fname = os.path.basename(filepath)
    return int(re.sub(r"[^0-9]", "", fname))

def get_image_files(dataset_filepath, exclude_images=None):
    if not os.path.isdir(dataset_filepath):
        raise Exception("Images filepath '%s' of the dataset seems to not " \
                        "exist or is not a directory." % (dataset_filepath,))

    images = []
    exclude_images = exclude_images if exclude_images is not None else set()
    exclude_filenames = set()
    for image_file in exclude_images:
        exclude_filenames.add(image_file.filename)

    for directory, subdirs, files in os.walk(dataset_filepath):
        for name in files:
            if re.match(r"^.*_[0-9]+\.(pgm|ppm|jpg|jpeg|png|bmp|tiff)$", name):
                if name not in exclude_filenames:
                    images.append(ImageFile(directory, name))
    images = sorted(images, key=lambda image: image.filename)
    return images

def get_image_pairs(dataset_filepath, nb_max, pairs_of_same_imgs=False,
                    ignore_order=True, exclude_images=None, seed=None,
                    verbose=False, input_shape=(32,32)):
    if seed is not None:
        state = random.getstate()
        random.seed(seed)

    # validate dataset directory
    if not os.path.isdir(dataset_filepath):
        raise Exception("Images filepath '%s' of the dataset seems to not " \
                        "exist or is not a directory." % (dataset_filepath,))

    # Build set of images to not use in image pairs
    exclude_images = exclude_images if exclude_images is not None else []
    exclude_images = set([img_pair.image1 for img_pair in exclude_images]
                         + [img_pair.image2 for img_pair in exclude_images])

    # load metadata of all images as ImageFile objects
    images = get_image_files(dataset_filepath, exclude_images=exclude_images)

    # mapping person=>images[]
    images_by_person = defaultdict(list)
    for image in images:
        images_by_person[image.person].append(image)

    nb_img = len(images)
    nb_people = len(images_by_person)

    # Create lists
    #  a) of all names of people appearing in the dataset
    #  b) of all names of people appearing in the dataset
    #     with at least 2 images
    names = []
    names_gte2 = []
    for person_name, images in images_by_person.items():
        names.append(person_name)
        if len(images) >= 2:
            names_gte2.append(person_name)

    # Calculate maximum amount of possible pairs of images showing thesame person
    if verbose:
        sum_avail_ordered = 0
        sum_avail_unordered = 0
        for name in names_gte2:
            k = len(images_by_person[name])
            sum_avail_ordered += k*(k-1)
            sum_avail_unordered += k*(k-1)/2
        print("Can collect max %d ordered and %d unordered pairs of images " \
              "that show the _same_ person." \
              % (sum_avail_ordered, sum_avail_unordered))

    # result
    pairs = []

    # counters
    nb_added = 0
    nb_same_p_same_img = 0 # pairs of images of same person, same image
    nb_same_p_diff_img = 0 # pairs of images of same person, different images
    nb_diff = 0

    # set that saves identifiers for pairs of images that have
    # already been added to the result.
    added = set()

    # -------------------------
    # y = 1 (pairs with images of the same person)
    # -------------------------
    while nb_added < nb_max // 2:
        # pick randomly two images and make an ImagePair out of them
        person = random.choice(names_gte2)
        image1 = random.choice(images_by_person[person])
        if pairs_of_same_imgs:
            image2 = random.choice(images_by_person[person])
        else:
            image2 = random.choice([image for image in \
                                    images_by_person[person] \
                                    if image != image1])

        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)

        # add the ImagePair to the output, if the same pair hasn't been already
        # picked
        if key not in added:
            pairs.append(pair)
            nb_added += 1
            nb_same_p_same_img += 1 if pair.same_image else 0
            nb_same_p_diff_img += 1 if not pair.same_image else 0

            added.add(key)

    # -------------------------
    # y = 0 (pairs with images of different persons)
    # -------------------------
    while nb_added < nb_max:
        # pick randomly two different persons names to sample each one image
        # from
        person1 = random.choice(names)
        person2 = random.choice([person for person in names \
                                 if person != person1])

        # we dont have to check here whether the images are the same,
        # because they come from different persons
        image1 = random.choice(images_by_person[person1])
        image2 = random.choice(images_by_person[person2])
        pair = ImagePair(image1, image2)
        key = pair.get_key(ignore_order)

        # add the ImagePair to the output, if the same pair hasn't been already
        # picked
        if key not in added:
            pairs.append(pair)
            nb_added += 1
            nb_diff += 1
            # log this pair as already added (dont add it a second time)
            added.add(key)

    # Shuffle the created list
    random.shuffle(pairs)

    # Print some statistics
    if verbose:
        print("Collected %d pairs of images total." % (nb_added,))
        print("Collected %d pairs of images showing the same person (%d are " \
              "pairs of identical images)." % \
                (nb_same_p_same_img + nb_same_p_diff_img, nb_same_p_same_img))
        print("Collected %d pairs of images showing different persons." \
                % (nb_diff,))

    # reset the RNG to the state that it had before calling the method
    if seed is not None:
        random.setstate(state) # state was set at the start of this function

    return pairs

def split_data_face(image_pairs, height, width):
    X = np.zeros((len(image_pairs), 2, height, width, IMAGE_CHANNELS), dtype=np.uint8)
    y = np.zeros((len(image_pairs),), dtype=np.float32)

    for i, pair in enumerate(image_pairs):
        X[i] = pair.get_contents(height, width)
        y[i] = Y_SAME if pair.same_person else Y_DIFFERENT

        wait = int(0 + (100 - 0) * (i - 0) / ((len(image_pairs)-1) - 0))
        print(" Splitting Dataset: %s%s" % ( wait, "%"), end="\r")

    return X, y


def split_data(X, y, validation_split=.2, setName=("Training Set", "Test Set")):
    num_train_samples = int((1 - validation_split)*len(X))
    X_train, y_train, X_test, y_test = X[:num_train_samples], y[:num_train_samples], X[num_train_samples:], y[num_train_samples:]
    print("\t %s: %s %s" % (setName[0], len(X_train), len(y_train)))
    print("\t %s: %s %s" % (setName[1], len(X_test), len(y_test)))
    return X_train, X_test, y_train, y_test

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'woman':0, 'man':1}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    else:
        raise Exception('Invalid dataset name')
