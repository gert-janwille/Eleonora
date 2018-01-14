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

reset_time = 10
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

PREDICT_ACC = 0.3

HOTKEYS = [
    './eleonora/data/hotkeys/nora.pmdl'
]

EXIT_WORDS = ['bedankt', 'tot ziens', 'salu', 'niets', 'sorry voor het storen']

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
