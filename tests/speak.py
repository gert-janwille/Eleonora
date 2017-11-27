from gtts import gTTS
import os
tts = gTTS(text='Hallo ik praat!', lang='nl')
tts.save("hallo.mp3")
os.system("mpg321 hallo.mp3")
