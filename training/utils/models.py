from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from utils.util import flow_batches
from keras.utils import generic_utils
from keras.layers import merge, Input
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Convolution2D, Dropout, Flatten, Conv2D

INPUT_CHANNELS = 1
INPUT_HEIGHT = 32
INPUT_WIDTH = 32

def conv(x, n_filters, k, s, border_same=True, drop=0.0, sigma=0.0):
    padding = "same" if border_same else "valid"
    x = Convolution2D(n_filters, k, strides=s, padding=padding, kernel_initializer="orthogonal")(x)
    x = LeakyReLU(0.33)(x)
    if sigma > 0:
        x = GaussianNoise(sigma)(x)
    if drop > 0:
        x = Dropout(drop)(x)
    return x


def create_model():

    img_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32")

    x = conv(img_input, 32, (3, 3), (1, 1), False) # 30x30
    x = conv(x, 32, (3, 3), (1, 1), False) # 28x28

    x = conv(x, 64, (5, 5), (2, 2), True) # 14x14
    x = conv(x, 64, (3, 3), (1, 1), False) # 12x12

    x = conv(x, 128, (5, 5), (2, 2), True) # 6x6
    x = conv(x, 128, (3, 3), (1, 1), False, drop=0.25) # 4x4

    x = Flatten()(x)
    x = Dense(512, kernel_initializer="glorot_normal")(x)

    x = BatchNormalization()(x)
    x = Activation("tanh")(x)


    face_model = Model(img_input, x)

    face_left_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),dtype="float32", name="face_left")
    face_right_input = Input(shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32", name="face_right")
    face_left = face_model(face_left_input)
    face_right = face_model(face_right_input)


    merged = layers.add([face_left, face_right])
    merged = GaussianNoise(0.5)(merged)
    merged = Dropout(0.2)(merged)

    merged = Dense(256, kernel_initializer="glorot_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = LeakyReLU(0.33)(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(1)(merged)
    merged = Activation("sigmoid")(merged)

    classification_model = Model(inputs=[face_left_input, face_right_input], outputs=merged)

    optimizer = Adam()

    print("Compiling model...")
    classification_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return classification_model, optimizer

def fit_model(identifier, model, optimizer, epoch_start, history, ia_train, ia_val, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE, BATCH_SIZE_VAL):
    print(BATCH_SIZE)
    for epoch in range(0, EPOCHS):
        print("Epoch", epoch, "/", EPOCHS)

        loss_train_sum = 0
        loss_val_sum = 0
        acc_train_sum = 0
        acc_val_sum = 0

        nb_examples_train = X_train.shape[0]
        nb_examples_val = X_val.shape[0]

        progbar = generic_utils.Progbar(nb_examples_train, interval=0)

        for X_batch, Y_batch in flow_batches(X_train, y_train, ia_train, batch_size=BATCH_SIZE, shuffle=True, train=True):

            bsize = X_batch[0].shape[0]

            loss, acc = model.train_on_batch(X_batch, Y_batch)

            progbar.add(bsize, values=[("train loss", loss), ("train acc", acc)])
            loss_train_sum += (loss * bsize)
            acc_train_sum += (acc * bsize)


        progbar = generic_utils.Progbar(nb_examples_val, interval=0)

        for X_batch, Y_batch in flow_batches(X_val, y_val, ia_val, batch_size=BATCH_SIZE_VAL, shuffle=False, train=False):
            bsize = X_batch[0].shape[0]

            loss, acc = model.test_on_batch(X_batch, Y_batch)
            progbar.add(bsize, values=[("val loss", loss), ("val acc", acc)])

            loss_val_sum += (loss * bsize)
            acc_val_sum += (acc * bsize)


        loss_train = loss_train_sum / nb_examples_train
        acc_train = acc_train_sum / nb_examples_train
        loss_val = loss_val_sum / nb_examples_val
        acc_val = acc_val_sum / nb_examples_val

        history.add(epoch, loss_train=loss_train, loss_val=loss_val, acc_train=acc_train, acc_val=acc_val)

    return model, history
