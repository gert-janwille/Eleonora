import os
# import asyncio
from gtts import gTTS
from playsound import playsound

# TODO: Class spreech
#   - initialize language
#   - simple hash and decrypte function
#   - hash the sentence needs to be said, save the has as filename
#     when Eleonora needs to say something, look in folder if hashes
#     are containing the sentence she needs to say
#   - Async speak and run other functions

class Speak(object):
    """docstring for Speak."""
    def __init__(self, arg):
        self.arg = arg

def listen():
    print("listening")

def welcome(*args):
    sentence, lang = args[0], args[1]
    tts = gTTS(text=sentence, lang=lang)
    tts.save("hallo.mp3")
    playsound('hallo.mp3')
    return
    # os.system("mpg321 hallo.mp3")
