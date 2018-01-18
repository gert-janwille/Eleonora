import numpy as np
from training.utils.History import History
from training.utils.util import modelToJSON
from training.libs.ImageAugmenter import ImageAugmenter
from training.models.models import facial_model, fit_facial_model
from training.utils.datasets import get_image_pairs, split_data_face

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_data_format('channels_first')

def facial_training():
    TRAIN_COUNT_EXAMPLES = 256 #4000
    VALIDATION_COUNT_EXAMPLES = 256
    EPOCHS = 1000
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 64
    VERBOSE = True
    SEED = 42
    INPUT_HEIGHT = INPUT_WIDTH = 32
    INPUT_SHAPE = (INPUT_WIDTH, INPUT_HEIGHT)

    imagePath = './training/data/lfwcrop_grey/faces'

    print("Loading validation dataset...")
    pairs_val = get_image_pairs(imagePath, VALIDATION_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=list(), seed=SEED, verbose=VERBOSE, input_shape=INPUT_SHAPE)

    # load training set
    print("\nLoading training dataset...")
    pairs_train = get_image_pairs(imagePath, TRAIN_COUNT_EXAMPLES, pairs_of_same_imgs=False, ignore_order=True, exclude_images=pairs_val, seed=SEED, verbose=VERBOSE, input_shape=INPUT_SHAPE)

    # check if more pairs have been requested than can be generated
    assert len(pairs_val) == VALIDATION_COUNT_EXAMPLES
    assert len(pairs_train) == TRAIN_COUNT_EXAMPLES

    print("Splitting Data...")
    X_val, y_val = split_data_face(pairs_val, height=INPUT_HEIGHT, width=INPUT_WIDTH)
    print("\n")
    X_train, y_train = split_data_face(pairs_train, height=INPUT_HEIGHT, width=INPUT_WIDTH)

    print("X Validation", len(X_val[0]))
    print("y Validation", y_val[0])


    # initialize the network
    print("Creating model...")
    model, optimizer = facial_model()

    # initialize the image augmenter for training images
    ia_train = ImageAugmenter(INPUT_WIDTH, INPUT_HEIGHT,
                                 hflip=True, vflip=False,
                                 scale_to_percent=1.1,
                                 scale_axis_equally=False,
                                 rotation_deg=20,
                                 shear_deg=6,
                                 translation_x_px=4,
                                 translation_y_px=4)

    ia_train.pregenerate_matrices(15000)
    ia_val = ImageAugmenter(INPUT_WIDTH, INPUT_HEIGHT)

    epoch_start = 0
    history = History()

    print("Model summary:")
    model.summary()

    model_json = model.to_json()
    with open("./training/models/facial_model.json", "w") as json_file:
        json_file.write(model_json)

    # run the training loop
    print("Training...")
    model, history = fit_facial_model('facialTraining', model, optimizer, epoch_start, history, ia_train, ia_val, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE, BATCH_SIZE_VAL)
    print("Done Training\n")

    print("Saving Model...")
    modelToJSON(model, history)

    print("Model Saved!")
