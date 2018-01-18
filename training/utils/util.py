import random
import datetime
import numpy as np
from keras.utils import np_utils

def preprocess_input(x):
    temp = []
    for idx, arr in enumerate(x):
        arr = np.array(arr).astype('float32')
        arr = arr / 255.0
        temp.append(arr)
    return temp

def to_categorical(classes, num_classes=2):
    temp = []
    for idx, arr in enumerate(classes):
        temp.append(np_utils.to_categorical(arr, num_classes))
    return temp

def modelToJSON(model, history):
    # serialize model to JSON
    model_json = model.to_json()
    i = datetime.datetime.now()
    t = "%s-%s-%s_%s-%s-%s" % (i.day, i.month, i.year,i.hour, i.month, i.second)

    #Save the history to a csv file
    history.save_to_filepath("./models/report_" + t + ".csv")

    with open("./models/trained_facial_model_" + t + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to H5
    model.save_weights("./models/trained_facial_model_" + t + ".h5")

def flow_batches(X_in, y_in, ia, batch_size=128, shuffle=False, train=False):
    if shuffle:
        X = np.copy(X_in)
        y = np.copy(y_in)

        seed = random.randint(1, 10e6)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
    else:
        X = X_in
        y = y_in

    nb_examples = X.shape[0]
    batch_start = 0
    while batch_start < nb_examples:
        batch_end = batch_start + batch_size
        if batch_end > nb_examples:
            batch_end = nb_examples

        batch = X[batch_start:batch_end]
        nb_examples_batch = batch.shape[0]

        # augment the images of the batch
        batch_img1 = batch[:, 0, ...] # left images
        batch_img2 = batch[:, 1, ...] # right images
        #if train:
        batch_img1 = ia.augment_batch(batch_img1)
        batch_img2 = ia.augment_batch(batch_img2)
        #else:
        #    batch_img1 = batch_img1 / 255.0
        #    batch_img2 = batch_img2 / 255.0

        # resize and merge the pairs of images to shape (B, 1, 32, 64), where
        # B is the size of this batch and 1 represents the only channel
        # of the image (grayscale).
        # X has usually shape (20000, 2, 64, 64, 3)
        height = X.shape[2]
        width = X.shape[3]
        #nb_channels = X.shape[4] if len(X.shape) == 5 else 1
        nb_channels = X.shape[4]
        X_batch_left = np.zeros((nb_examples_batch, height, width,
                                nb_channels))
        X_batch_right = np.zeros((nb_examples_batch, height, width,
                                 nb_channels))
        #X_batch_left = np.zeros((nb_examples_batch, height, width))
        #X_batch_right = np.zeros((nb_examples_batch, height, width))
        for i in range(nb_examples_batch):
            # sometimes switch positions (left/right) of images during training
            if train and random.random() < 0.5:
                img1 = batch_img2[i]
                img2 = batch_img1[i]
            else:
                img1 = batch_img1[i]
                img2 = batch_img2[i]

            #X_batch_left[i] = img1[:, :, np.newaxis]
            #X_batch_right[i] = img2[:, :, np.newaxis]
            X_batch_left[i] = img1
            X_batch_right[i] = img2

        # Collect the y values of the batch
        y_batch = y[batch_start:batch_end]

        # from (B, H, W, C) to (B, C, H, W)
        X_batch_left = X_batch_left.transpose(0, 3, 1, 2)
        X_batch_right = X_batch_right.transpose(0, 3, 1, 2)

        yield [X_batch_left, X_batch_right], y_batch
        batch_start = batch_start + nb_examples_batch
