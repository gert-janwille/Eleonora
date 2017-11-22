from eleonora.common.constants import *

def ask(q):
    if input("[" + R + "!" + W + "] " + q + "? (y/[N]) ") == "y":
        return True
    return False

def message(m):
    print("[" + T + "*" + W + "] " + m + "\n")

def warning(m):
    print("[" + T + "+" + W + "] " + m + "\n")

def header(headline):
    print(B + T + '\nEleonora' + W + " - " + headline + "\n\n")
