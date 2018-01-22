import json, requests
from eleonora.modules import UI
from playsound import playsound
import eleonora.utils.config as config

class GetLatestNews(object):
    """docstring for GetLatestNews."""
    def __init__(self):
        self.data = json.loads(requests.get(url=config.NEWS_URL).text)
        self.maxart = len(self.data['articles'])-1

        UI.news(async=True, manual=True)
        self.playFile('news.wav', 'news/')

    def generate(self):
        say = ''
        for i, art in enumerate(self.data['articles']):
            say = say + art['title']
            if i < self.maxart:
                say = say + ' en '
        return say

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)
