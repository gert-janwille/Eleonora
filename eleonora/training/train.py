import os
import time
import eleonora.common.ui as ui
from eleonora.common.constants import *
from eleonora.training.modules.train_emotion import train as trainEmotion
from eleonora.training.modules.train_gender import train as trainGender

trainings = {
    0: ["Emotion Training", trainEmotion],
    1: ["Gender Training", trainGender]
}

def main(args):
    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting TRAINING Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    # Print list of available trainings
    print("\n "+B+T+"Available Trainings:"+W)
    for idx, value in trainings.items():
        print("\t%s%s%s: %s" %(T,idx, W,value[0]))

    # Ask the input
    nrTraining = int(input("\nEnter Number > "+T+B))
    print(W)
    trainings[nrTraining][1]()
