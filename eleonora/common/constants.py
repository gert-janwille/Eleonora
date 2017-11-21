#                               __                    __
#                              /\ \__                /\ \__
#   ___    ___     ___     ____\ \ ,_\    __      ___\ \ ,_\   ____
#  /'___\ / __`\ /' _ `\  /',__\\ \ \/  /'__`\  /' _ `\ \ \/  /',__\
# /\ \__//\ \L\ \/\ \/\ \/\__, `\\ \ \_/\ \L\.\_/\ \/\ \ \ \_/\__, `\
# \ \____\ \____/\ \_\ \_\/\____/ \ \__\ \__/.\_\ \_\ \_\ \__\/\____/
#  \/____/\/___/  \/_/\/_/\/___/   \/__/\/__/\/_/\/_/\/_/\/__/\/___/  .txt
#
#

import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

VERSION = "BETA 0.0.1"

NUMBER_OF_DETECTION = 5
FACE_CASCADE_FILE = os.path.join('./eleonora/data/haarcascades/','haarcascade_frontalface_alt.xml')


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


def showImages(amount, arr, larr):
    fig = plt.figure(figsize=(16, 9))
    for i in range(0,amount):
        ax = fig.add_subplot(2, 5, i+1)
        imshow(arr[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.axis('off')
        plt.legend('off')

        ax.set_title(larr[i])
        # x and y axis should be equal length
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
    plt.show()
