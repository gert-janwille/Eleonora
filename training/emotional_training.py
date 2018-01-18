import os
import numpy as np
import pandas as pd
from training.models.models import *
from training.utils.datasets import DataManager, split_data
from training.utils.util import preprocess_input, to_categorical
from training.utils.constants import *

from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def modelToJSON(model, filename="model"):
    # serialize model to JSON
    modelName = str(models[architecture][1])
    model_json = model.to_json()

    if not os.path.exists("./models/" + modelName):
        os.makedirs("./models/" + modelName)

    if not os.path.exists("./models/" + modelName + "/" + dataset_name):
        os.makedirs("./models/" + modelName + "/" +dataset_name)

    with open("./models/" + modelName + "/" + dataset_name + "/model_" + filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to H5
    model.save_weights("./models/" + modelName + "/" + dataset_name + "/model_" + filename + ".h5")


def emotional_training():
    # parameters
    db_items = False #False
    dense = 16 #16
    batch_size = 32 #32

    # epochs = 50 #1000
    epochs = int(input("Epochs > "))

    validation_split = .2
    num_classes = 7
    patience = 50
    base_path = './training/models/'

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)

    # Choose Dataset
    print("\n 0: fer2013\n 1: KDEF\n")
    datasets = {
        0: ["fer2013", (48, 48, 1)],
        1: ["KDEF", (64, 64, 1)]
    }

    c_data = int(input("Dataset Number > "))
    print(c_data, datasets[c_data][0], datasets[c_data][1])
    dataset_name = datasets[c_data][0]
    input_shape = datasets[c_data][1]

    # model parameters/compilation
    print(" 0: simple_CNN,\n 1: simpler_CNN,\n 2: simple_big_CNN,\n 3: max_CNN,\n 4: mini_XCEPTION,\n 5: tiny_XCEPTION,\n 6: big_XCEPTION\n")
    models = {
        0: [simple_CNN, "simple_CNN"],
        1: [simpler_CNN, "simpler_CNN"],
        2: [medium_CNN, "medium_CNN"],
        3: [max_CNN, "max_CNN"],
        4: [mini_XCEPTION, "mini_XCEPTION"],
        5: [tiny_XCEPTION, "tiny_XCEPTION"],
        6: [big_XCEPTION, "big_XCEPTION"]
    }

    architecture = int(input("Model number > "))
    model = models[architecture][0](input_shape, num_classes, d=dense)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()



    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, labels = data_loader.get_data(db_items)

    training_images, test_images, y_train, y_test = split_data(faces, labels, validation_split)

    training_images, test_images = preprocess_input((training_images, test_images))
    y_train, y_test = to_categorical((y_train, y_test), num_classes)

    # Split data into test and validate set
    val_images, test_images, y_val, y_test = split_data(test_images, y_test, .5, ("Validation Set", "Test Set"))

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


    model.fit_generator(data_generator.flow(training_images, y_train,
                        batch_size=batch_size),
                        steps_per_epoch=len(training_images) / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(val_images, y_val))


    # Print the accuracy score
    score = model.evaluate(test_images, y_test, verbose=1)
    print("[" + T + "!" + W + "] Accuraty Score TEST Set of: " + B + str(score[1]*100)+ "%" + W + "\n")
    testscore = score

    score = model.evaluate(training_images, y_train, verbose=1)
    print("[" + T + "!" + W + "] Accuraty Score TRAIN Set of: " + B + str(score[1]*100)+ "%" + W + "\n")

    name = str(int(testscore[1]*100))
    modelToJSON(model, name)
