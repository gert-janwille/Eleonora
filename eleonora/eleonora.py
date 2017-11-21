import sys
import time
import eleonora.common.ui as ui
from eleonora.common.constants import *
import eleonora.modules.detectFaces as df

def main(args):
    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    # Start Looking for faces
    df.detectFaces()
