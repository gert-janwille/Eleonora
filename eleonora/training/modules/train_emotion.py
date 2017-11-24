from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from eleonora.common.constants import *
from eleonora.utils.input import ask, message, header
from eleonora.utils.datasets import DataManager, split_data
from eleonora.training.models.cnn import max_CNN, modelToJSON
from eleonora.utils.preprocessor import preprocess_input, to_categorical


def train():
    header("Emotional Training with a Convolutional Neural Network")
    # TODO: Training max
    # - Tessting max CNN
    # - probably using simpler_CNN (16/32/64)
    # parameters
    dataset_name = 'fer2013'
    dense = 32
    batch_size = 32 #32
    epochs = 100 #1000

    img_rows, img_cols = 48, 48
    input_shape = (img_rows, img_cols, 1)

    validation_split = .3
    num_classes = 7
    patience = 50

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)

    # model parameters/compilation
    model = max_CNN(input_shape, num_classes, d=dense)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if ask("Network Overview"):
        model.summary()

    # Load Dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, labels = data_loader.get_data()

    training_images, test_images, y_train, y_test = split_data(faces, labels, validation_split)

    training_images, test_images = preprocess_input((training_images, test_images))
    y_train, y_test = to_categorical((y_train, y_test), num_classes)

    # Split data into test and validate set
    val_images, test_images, y_val, y_test = split_data(test_images, y_test, .5, ("Validation Set", "Test Set"))

    # callbacks
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)

    callbacks = [early_stop, reduce_lr]


    if ask("Start Training"):
        # Train the CNN
        message("Start Training")

        model.fit_generator(data_generator.flow(training_images, y_train,
                            batch_size=batch_size),
                            steps_per_epoch=len(training_images) / batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(val_images, y_val))


        message("Done Training")

        # Print the accuracy score
        score = model.evaluate(test_images, y_test, verbose=1)
        print("[" + T + "!" + W + "] Accuraty Score TEST Set of: " + B + str(score[1]*100)+ "%" + W + "\n")

        score = model.evaluate(training_images, y_train, verbose=1)
        print("[" + T + "!" + W + "] Accuraty Score of: " + B + str(score[1]*100)+ "%" + W + "\n")

        name = str(int(score[1]*100))
        if ask("Save Model"):
            modelToJSON(model, name)
