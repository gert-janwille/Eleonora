import json, requests
from eleonora.modules import UI
from playsound import playsound
import eleonora.utils.config as config

class GetWeather(object):
    """docstring for GetWeather."""
    def __init__(self):
        self.data = json.loads(requests.get(url=config.WEATHER_URL).text)
        UI.weather(async=True, manual=True)

    def generate(self):
        say = ''
        currentTemp = str(self.data['main']['temp'])
        maxTemp = str(self.data['main']['temp_max'])
        description = str(self.data['weather'][0]['description'])

        say = 'Het is nu ' + description + ' en ' + currentTemp + ' graden, het word maximum ' + maxTemp
        return say

    def playFile(self, audio, folder=False):
        if not folder:
            folder = ''
        playsound(config.AUDIO_PREFIX + folder + audio)
