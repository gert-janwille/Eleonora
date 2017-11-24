from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from eleonora.common.constants import *
from eleonora.utils.input import ask, message, header
from eleonora.utils.datasets import DataManager, split_data
from eleonora.training.models.cnn import simpler_CNN, modelToJSON
from eleonora.utils.preprocessor import preprocess_input, to_categorical


def train():
    header("Gender Training with a Convolutional Neural Network")
