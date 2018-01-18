import playsound
from eleonora.modules import UI

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

        print('Starting Mindfulness...')

        # TODO: Mindfulness
        #       - Eleonora introduce and say what to do
        #       - user can see circle who becomes bigger for inhaling
        #        and smaller when exhaling
        #       - Eleonora thanks for doing mindfulness

        UI.mindfulness(async=True) # if async, can play audio dont forget to pause time.sleep()

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)
