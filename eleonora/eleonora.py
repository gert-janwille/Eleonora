import time
import eleonora.common.ui as ui
from eleonora.common.constants import *
from eleonora.modules.detectFaces import detectFaces

def main():
    ui.clearScreen()

    print ('[' + T + '*' + W + '] Starting Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    print ('[' + T + '+' + W + '] Looking for faces')
    detectFaces()
