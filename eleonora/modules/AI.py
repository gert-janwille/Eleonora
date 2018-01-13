import os
import random
import threading
import numpy as np
from gtts import gTTS
from scipy.io import loadmat
import speech_recognition as sr
from playsound import playsound
import eleonora.utils.config as config
from eleonora.utils.input import message, warning
from eleonora.utils.util import getVerifyFile
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
            t = threading.Thread(target=self.worker, args=(fvec1, verifyFrame, obj))
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
    data = loadmat(path, matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    description = data['meta'][0,0].classes[0,0].description

    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3)

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


class HotKeyListener(object):
    def __init__(self, hotkeys=config.HOTKEYS, sensitivity=.5, audio_gain=1):
        self.hotkeys = hotkeys
        self.sensitivity = sensitivity
        self.audio_gain = audio_gain

    def listener(self):
        message('Start Detecting Hotkeys')
        detector = snowboydecoder.HotwordDetector(self.hotkeys, sensitivity=self.sensitivity, audio_gain=self.audio_gain)
        self.detector = detector
        detector.start(self.callback)

    def listen(self, callback):
        self.callback = callback
        thread_listener = threading.Thread(target=self.listener)
        thread_listener.start()

    def stop(self):
        self.detector.terminate()


class SpeechRecognition(object):
    def __init__(self, lang='en-us'):
        self.lang = lang

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
            playsound(config.AUDIO_PREFIX + 'response/ja.wav')
            self.talk(name)
            playsound(config.AUDIO_PREFIX + 'response/nameResponce.wav')
        else:
            playsound(config.AUDIO_PREFIX + 'response/generalResponce.wav')

    def welcomePerson(self, name):
        files = getFiles('welcomePerson')

        playsound(config.AUDIO_PREFIX + 'welcome/hallo.wav')
        self.talk(name)
        playsound(config.AUDIO_PREFIX + 'welcome/' + random.choice(files))
        playsound(config.AUDIO_PREFIX + 'response/nameResponce.wav')


    def ping(self, high=False):
        if high:
            f = 'ding.wav'
        else:
            f = 'dong.wav'
        playsound(config.AUDIO_PREFIX + f)

    def tts(self, audio, r):
        oeps = getFiles('oeps')
        try:
            data = r.recognize_google(audio, language="nl-BE").lower()
            print("You said: " + data)

            # TODO: process data
            if data in config.EXIT_WORDS:
                return True
            elif data in ['open je deur', 'open jouw deur']:
                warning('Opening the door may lead to a vulnerability!')
                playsound(config.AUDIO_PREFIX + 'error/' + 'danger_0.wav')
            else:
                return False
            return False

        except sr.UnknownValueError:
            playsound(config.AUDIO_PREFIX + 'error/' + random.choice(oeps))
        except sr.RequestError as e:
            playsound(config.AUDIO_PREFIX + 'error/' + random.choice(oeps))

    def listen(self):
        w = getFiles('yourewelcome')
        r = sr.Recognizer()
        hasToQuit = False

        with sr.Microphone() as source:
            while not hasToQuit:
                hasToQuit = self.tts(r.listen(source), r)
        playsound(config.AUDIO_PREFIX + 'thanks/' + random.choice(w))

def getFiles(key):
    s = []
    path = './eleonora/data/wav/'
    for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.startswith(key):
                s.append(filename)
    return s
