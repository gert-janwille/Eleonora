import random
import numpy as np
from playsound import playsound
import eleonora.utils.config as config
from eleonora.interact.Mindfulness import *
from eleonora.utils.input import message, warning, userInput

class Emotion(object):
    def __init__(self, emotion, speech=None):
        self.emotion = emotion
        self.speech = speech

        # function_list = self.angry_interaction()
        if emotion == 'angry':
            function_list = self.angry_interaction()
        elif emotion == 'disgusted':
            function_list = self.disgusted_interaction()
        elif emotion == 'fearful':
            function_list = self.fearful_interaction()
        elif emotion == 'happy':
            function_list = self.happy_interaction()
        elif emotion == 'sad':
            function_list = self.sad_interaction()
        elif emotion == 'surprised':
            function_list = self.surprised_interaction()
        elif emotion == 'neutral':
            function_list = self.neutral_interaction()
        elif emotion == None:
            function_list = self.all_interactions()
        else:
            warning('Something went wrong')
            return


        # Pick an function
        opt = random.choice(function_list)
        # Read all Function
        self.tellFunction(opt['name'])
        # Ask if user want it
        if self.getAnswer():
            option = opt['func'](self.speech)

    def all_interactions(self):
        all_array = np.concatenate(
        (
            self.angry_interaction(),
            self.disgusted_interaction(),
            self.fearful_interaction(),
            self.happy_interaction(),
            self.sad_interaction(),
            self.surprised_interaction(),
            self.neutral_interaction()
        ), axis=0)
        return all_array

    def angry_interaction(self):
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR

    def disgusted_interaction(self):
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR

    def fearful_interaction(self):
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR

    def happy_interaction(self):
        # Jokes
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR
    def sad_interaction(self):
        # Give Hug
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR

    def surprised_interaction(self):
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR

    def neutral_interaction(self):
        # Jokes
        ARR = [
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ARR


    def getAnswer(self):
        message('Answer Yes or No')
        data = self.speech.getShort()

        if data == None:
            self.getAnswer()
            return False
        if 'ja' in data.split(' '):
            return True
        elif 'nee' in data.split(' '):
            return False

    def tellFunction(self, f):
        self.playFile(f + '.wav','functions/')

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)
