import time
from playsound import playsound
from eleonora.modules import UI
import eleonora.utils.config as config
from eleonora.utils.input import header

class Mindfulness(object):
    """
    Eleonora will tell the user to bread deep in and out,
    user wil hear ping when done and visuals for in and exhaling.
    This will make the heart rate go low so the user will
    reduce his/her stresslevel
    """
    def __init__(self, speech=''):
        self.speech = speech
        self.folder = 'mindfulness/'

        header('Starting Mindfulness...')
        self.playFile('mindfulness_0.wav', self.folder)
        time.sleep(1)
        self.playFile('mindfulness_1.wav', self.folder)

        UI.mindfulness(async=True)

        self.playFile('mindfulness_2.wav', self.folder)
        time.sleep(5)
        self.playFile('mindfulness_3.wav', self.folder)

        time.sleep(30)
        UI.happy(async=True)
        self.playFile('done.wav', self.folder)

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)
