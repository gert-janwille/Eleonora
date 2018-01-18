import random
import eleonora.utils.config as config
from eleonora.interact.Mindfulness import *
from eleonora.utils.input import message, warning, userInput

class Emotion(object):
    def __init__(self, emotion, speech):
        self.speech = speech
        self.emotion = emotion
        self.playFile = speech.playFile

        function_list = self.angry_interaction()
        # if emotion == 'angry':
        #     function_list = self.angry_interaction()
        # elif emotion == 'disgusted':
        #     function_list = self.disgusted_interaction()
        # elif emotion == 'fearful':
        #     function_list = self.fearful_interaction()
        # elif emotion == 'happy':
        #     function_list = self.happy_interaction()
        # elif emotion == 'sad':
        #     function_list = self.sad_interaction()
        # elif emotion == 'surprised':
        #     function_list = self.surprised_interaction()
        # elif emotion == 'neutral':
        #     function_list = self.neutral_interaction()
        # else:
        #     warning('Something went wrong')
        #     return

        # Pick an function
        opt = random.choice(function_list)
        # Read all Function
        self.tellFunction(opt['name'])
        # Ask if user want it
        if self.getAnswer():
            option = opt['func'](self.speech)

    def angry_interaction(self):
        ANGRY_FUNCTIONS = [
            {"name": 'mindfulness', "func": Mindfulness},
            {"name": 'mindfulness', "func": Mindfulness}
        ]
        return ANGRY_FUNCTIONS

    def disgusted_interaction(self):
        print('disgusted')

    def fearful_interaction(self):
        print('fearful')

    def happy_interaction(self):
        # Jokes
        print('happy')

    def sad_interaction(self):
        # Give Hug
        print('sad')

    def surprised_interaction(self):
        print('surprised')

    def neutral_interaction(self):
        # Jokes
        print('neutral')


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
        # c = len(arr)
        # for i, key in enumerate(arr):
        #     if i == (c-1) and c != 1:
        #         self.playFile('of.wav')
        #     self.playFile(key + '.wav','functions/')
