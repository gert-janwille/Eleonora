import time
import threading
import eleonora.utils.config as config

def look(async=False):
    config.emitter = 'look'
    waitandreset(30, async)

def love(async=False):
    config.emitter = 'love'
    waitandreset(5, async)

def happy(async=False):
    config.emitter = 'happy'
    waitandreset(5, async)

def mindfulness(async=False):
    config.emitter = 'mindfulness'
    waitandreset(30, async)

def news(async=False, manual=False):
    config.emitter = 'news'
    waitandreset(3, async, manual)

def weather(async=False, manual=False):
    config.emitter = 'weather'
    waitandreset(3, async, manual)

def reset():
    config.emitter = 'reset'

def waitandreset(seconds, async, manual= False):
    if async:
        time_thread = threading.Thread(name='waitadnreset', target=startTimer, args=(seconds, manual,))
        time_thread.start()
    else:
        startTimer(seconds)

def startTimer(seconds, manual):
    time.sleep(seconds)
    if not manual:
        config.emitter = 'reset'
