from training.emotional_training import emotional_training
from training.facial_training import facial_training

def train():
    print('\n0: Emotional Training')
    print('1: Facial Training\n')

    choose = int(input("Type Number > "))

    if choose == 0:
        emotional_training()

    if choose == 1:
        facial_training()
