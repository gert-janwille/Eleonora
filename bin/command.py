#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess


ENVR = "aibot"


args = sys.argv[1:]

def main():
    options = {
        "activate": activate,
        "deactivate": deactivate,
        "save": savePackages
    }
    
    options[args[0]]()
    

def savePackages():
    os.system("pip freeze > requirements.txt")
    
def activate():
    print("source activate %s" % ENVR)
    os.system("source activate %s" % ENVR)
    

def deactivate():
    print("source deactivate %s" % ENVR)
    os.system("source deactivate %s" % ENVR)
    
if __name__ == '__main__':
    main()