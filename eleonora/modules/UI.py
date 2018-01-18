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

def waitandreset(seconds, async):
    if async:
        time_thread = threading.Thread(name='waitadnreset', target=startTimer, args=(seconds,))
        time_thread.start()
    else:
        startTimer(seconds)

def startTimer(seconds):
    time.sleep(seconds)
    config.emitter = 'reset'
