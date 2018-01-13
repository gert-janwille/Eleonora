import eleonora.utils.config as config

def ask(q):
    if input("[" + config.R + "!" + config.W + "] " + q + "? (y/[N]) ") == "y":
        return True
    return False

def message(m):
    print("[" + config.T + "*" + config.W + "] " + m + "\n")

def warning(m):
    print("[" + config.R + "!" + config.W + "] WARNING: " + m + "\n")

def header(headline):
    print(config.B + config.T + '\nEleonora' + config.W + " - " + headline + "\n\n")

def quit(interface = False, value = ""):
    if not interface:
        return value == "quit" or value == "q"
    else:
        interface.waitKey(1) & 0xFF == ord('q')
