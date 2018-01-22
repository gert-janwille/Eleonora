#                               __                    __
#                              /\ \__                /\ \__
#   ___    ___     ___     ____\ \ ,_\    __      ___\ \ ,_\   ____
#  /'___\ / __`\ /' _ `\  /',__\\ \ \/  /'__`\  /' _ `\ \ \/  /',__\
# /\ \__//\ \L\ \/\ \/\ \/\__, `\\ \ \_/\ \L\.\_/\ \/\ \ \ \_/\__, `\
# \ \____\ \____/\ \_\ \_\/\____/ \ \__\ \__/.\_\ \_\ \_\ \__\/\____/
#  \/____/\/___/  \/_/\/_/\/___/   \/__/\/__/\/_/\/_/\/_/\/__/\/___/
#
#

VERSION = '1.0.0'

MOMENTUM = 5
VERBOSE = 1
width = height = 64
process_this_frame = True
number_of_detection = 5
active_mode = False

reset_time = 21
scaned_person = []

# starting lists for calculating modes
screens = []

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)
CUTSIZES = (18, 18)

AUDIO_PREFIX = './eleonora/data/wav/'
AUDIO_PATH = './eleonora/data/wav/speech.wav'

database_path = './eleonora/data/database.json'

facial_json_path_mat = './eleonora/data/models/facial_recognition.json'
facial_math_path = './eleonora/data/models/facial_recognition.mat'
emotion_model_path = './eleonora/data/models/fer2013_mini_XCEPTION.102-0.66.hdf5'
detection_model_path = './eleonora/data/haarcascades/haarcascade_frontalface_default.xml'

WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather?q=waregem,be&units=metric&&lang=nl&appid=369338237727f976d23c30c176a119bc"
NEWS_URL = "https://newsapi.org/v2/everything?sources=rtl-nieuws&apiKey=422359e512d8402b9ef19eee25de9bb0"

PREDICT_ACC = 0.3
emitter = ''
preEm = ''
MILLIS = 1000
HOTKEYS = [
    './eleonora/data/hotkeys/nora.pmdl'
]

EXIT_WORDS = ['bedankt', 'tot ziens', 'salu', 'niets', 'sorry voor het storen']
RANDOM_ACTIVITY_COMMANDS = ['ik verveel me', 'kan je iets voorstellen', 'wat kan ik doen', 'heb je suggesties', 'zoek een activiteit']
BACKDOOR_COMMANDS = ['open je deur', 'open jouw deur']
WEATHER_COMMANDS = ['wat is het weer vandaag', 'wat zou het weer zijn', 'welk weer is het', 'wat is het weer']
NEWS_COMMANDS = ['is er nieuws', 'wat is het nieuws', 'wat is het nieuws vandaag', 'wat is het nieuws van vandaag', 'is er nieuws beschikbaar']
# Console colors
W = '\033[0m'    # white (normal)
R = '\033[31m'   # red
G = '\033[32m'   # green
O = '\033[33m'   # orange
B = '\033[34m'   # blue
P = '\033[35m'   # purple
C = '\033[36m'   # cyan
GR = '\033[37m'  # gray
T = '\033[93m'   # tan
B = '\033[1m' # bold
