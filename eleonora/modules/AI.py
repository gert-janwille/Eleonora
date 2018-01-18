import random
import threading
import numpy as np
from gtts import gTTS
from scipy.io import loadmat
import speech_recognition as sr
from playsound import playsound
from eleonora.modules import Interact
import eleonora.utils.config as config
from eleonora.utils.util import getVerifyFile, getFiles
from eleonora.utils.input import message, warning, userInput
from scipy.spatial.distance import cosine as dcos
from eleonora.modules.snowboy import snowboydecoder

class Emotion_Recognizer(object):
    def __init__(self, model):
        self.model = model
        self.labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def predict(self, frame):
        emotion_prediction = self.model.predict(frame)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.labels[emotion_label_arg]
        return emotion_text

    def interact(self, speech, emotion):
        self.speech = speech
        self.playFile = speech.playFile

        # Say emotion
        sayEmotion_thread = threading.Thread(name='say emotion', target=self.sayEmotion, args=(emotion,))
        sayEmotion_thread.start()

        # Join the main Thread before next saying
        sayEmotion_thread.join()
        emotional_interaction = Interact.Emotion(emotion, self.speech)


    def sayEmotion(self, emotion):
        self.playFile('emotional_state_0.wav', 'emotional/')
        self.playFile(emotion + '.wav', 'emotional/emotions/')
        self.playFile('emotional_state_1.wav', 'emotional/')


class Facial_Recognizer(object):
    def __init__(self, model, sizes=(32,32)):
        self.model = model
        self.sizes = sizes

        self.predictions = []
        self.threads = []

    def verify(self, frame, db=None, key='face'):
        # Predict the face
        fvec1 = self.predict(frame)

        # Loop over db, start new thread(worker)
        for (i, obj) in enumerate(db):
            verifyFrame = getVerifyFile(obj[key], (self.sizes), prefix='./eleonora/data/faces/')
            t = threading.Thread(name='Verify Faces', target=self.worker, args=(fvec1, verifyFrame, obj))
            self.threads.append(t)
            t.start()

        # Join all threads - wait for all threads
        for x in self.threads:
            x.join()

        # Return false if no predictions
        if len(self.predictions) <= 0:
            return False

        # Get max accuracy from predictions array
        person = max(self.predictions, key=lambda ev: ev['acc'])
        return person['obj']

    def predict(self, f):
        return self.model.predict(f)[0,:]

    def worker(self, fvec1, verifyFrame, obj):
        # Predict vector from frame
        fvec2 = self.predict(verifyFrame)

        # Calculate the cosine similarity
        acc = dcos(fvec1, fvec2)

        if config.VERBOSE:
            print(obj['first_name'], acc)

        # Add object to array if more then config accuracy
        if acc < config.PREDICT_ACC:
            self.predictions.append({
                "obj": obj,
                "acc": acc
            })

def load_weights(kmodel, path):
    message('Finding Models...')
    data = loadmat(path, matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    description = data['meta'][0,0].classes[0,0].description

    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3)
    c = 0
    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            assert(f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert(l_bias.shape[1] == 1)
            assert(l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert(len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
            wait = int(0 + (100 - 0) * (c - 0) / ((l.shape[1] - 1) - 0))
            print("Loading Models: %s%s%s%s" % (config.T, wait, "%", config.W), end="\r")
            c = c + 1


class HotKeyListener(object):
    def __init__(self, hotkeys=config.HOTKEYS, sensitivity=.5, audio_gain=1):
        self.hotkeys = hotkeys
        self.sensitivity = sensitivity
        self.audio_gain = audio_gain

    def listener(self):
        message('Start Detecting Hotkeys')
        self.detector = snowboydecoder.HotwordDetector(self.hotkeys, sensitivity=self.sensitivity, audio_gain=self.audio_gain)
        self.detector.start(self.callback)

    def listen(self, callback):
        self.callback = callback
        self.thread_listener = threading.Thread(name='HotKeyListener', target=self.listener)
        self.thread_listener.setDaemon(True)
        self.thread_listener.start()

    def pause(self):
        self.detector.terminate()

    def start(self):
        self.listener()


class SpeechRecognition(object):
    def __init__(self, lang='en-us'):
        self.lang = lang
        self.path = './eleonora/data/wav/'

    def tts(self, audio, r, option=''):
        oeps = getFiles('oeps', self.path)
        try:
            # Get Text from Speech & Print
            data = r.recognize_google(audio, language="nl-BE").lower()

            if config.VERBOSE:
                userInput(data)

            # If there is an option run first and stop
            if option == 'returndata':
                return data

            # TODO: process data & split in functions
            # All commands, return true when stop
            if data in config.EXIT_WORDS:
                return True
            elif data in ['nora', 'eleonora']:
                self.recall()
            elif data in config.BACKDOOR_COMMANDS:
                self.openBackdoor()
            else:
                return False
            return False

        # Do something on an Error
        except sr.UnknownValueError:
            self.playFile(random.choice(oeps), 'error/')
        except sr.RequestError as e:
            self.playFile(random.choice(oeps), 'error/')

    def talk(self, text):
        gTTS(text=text, lang=self.lang).save(config.AUDIO_PATH)
        playsound(config.AUDIO_PATH)

    def welcome(self):
        playsound(config.AUDIO_PREFIX + 'welcome/welcome.wav')

    def response(self):
        try:
            name = config.scaned_person['first_name']
        except Exception:
            name = False

        if name:
            self.playFile('ja.wav', 'response/')
            self.talk(name)
            self.playFile('nameResponce.wav', 'response/')
        else:
            self.playFile('generalResponce.wav', 'response/')

    def welcomePerson(self, name):
        files = getFiles('welcomePerson', self.path)

        self.playFile('hallo.wav', 'welcome/')
        self.talk(name)
        self.playFile(random.choice(files), 'welcome/')

    def ping(self, high=False):
        if high:
            f = 'ding.wav'
        else:
            f = 'dong.wav'
        self.playFile(f)

    def listen(self):
        w = getFiles('yourewelcome', self.path)
        r = sr.Recognizer()
        hasToQuit = False

        with sr.Microphone() as source:
            while not hasToQuit:
                hasToQuit = self.tts(r.listen(source), r)
        self.playFile(random.choice(w), 'thanks/')
        return True

    def getName(self):
        intro = getFiles('introducing', self.path)
        self.playFile(random.choice(intro), 'introducing/')
        r = sr.Recognizer()
        with sr.Microphone() as source:
            name = self.tts(r.listen(source), r, option='returndata')
        return name

    def getShort(self):
        r = sr.Recognizer()
        self.ping()
        with sr.Microphone() as source:
            data = self.tts(r.listen(source), r, option='returndata')
            print(data,'o')
        return data

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)


    # FUNCTIONS OF SPEECH
    def recall(self):
        self.playFile('ja.wav', 'response/')
        self.playFile('generalResponce.wav', 'response/')

    def openBackdoor(self):
        # TODO: Open Backdoor - use new class
        warning('Opening the door may lead to a vulnerability!')
        self.playFile('danger_0.wav', 'error/')
