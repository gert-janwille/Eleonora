import eleonora.utils.config as config

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

def quit(interface = False, value = ""):
    if not interface:
        return value == "quit" or value == "q"
    else:
        interface.waitKey(1) & 0xFF == ord('q')
