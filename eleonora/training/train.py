import os
import time
import eleonora.common.ui as ui
from eleonora.common.constants import *
from eleonora.training.train_emotion import train as trainEmotion

trainings = {
    0: ["Emotion Training", trainEmotion]
}

def main(args):
    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting TRAINING Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    # Print list of available trainings
    print("\n "+B+T+"Available Trainings:"+W)
    for idx, value in trainings.items():
        print("\t%s%s%s: %s" %(T,idx, W,value[0]))

    nrTraining = int(input("\nEnter Number > "+T+B))
    print(W)
    trainings[nrTraining][1]()
