import os
import time
import eleonora.common.ui as ui
from eleonora.common.constants import *
from eleonora.utils.input import header
from eleonora.training.train_emotion import train as trainEmotions

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["KERAS_BACKEND"] = "tensorflow"

def main(args):
    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting TRAINING Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    header("Emotional Training with a Convolutional Neural Network")
    trainEmotions()
