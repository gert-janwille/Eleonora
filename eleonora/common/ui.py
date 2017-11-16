"""
This module was made to handle the ui of the app
"""
import os

def clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')
