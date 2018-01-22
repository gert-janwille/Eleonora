import cv2
import PIL
import threading
import tkinter as tk
from PIL import Image, ImageTk
import eleonora.utils.config as config
from eleonora.modules import AI, DB, Engine
from eleonora.utils.input import quit, message, header

class Application:
    def __init__(self):
        self.prefix = './eleonora/data/video/'
        self.current_image = None
        self.loop = False
        self.end = False

        self.root = tk.Tk()
        self.root.title("Eleonora")
        self.panel = tk.Label(self.root)
        self.panel.pack(padx=0, pady=0)

        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.config(cursor="none")
        self.root.attributes("-fullscreen",True)

        self.setStream('happy_face.mp4')
        self.playVideo()

        self.check_Mode()

    def check_Mode(self):
        self.root.after(config.MILLIS, self.check_Mode)

        if config.preEm == config.emitter:
            return
        else:
            self.vs.release()

            if config.emitter == 'happy':
                self.setStream('happy_face.mp4')
            elif config.emitter == 'look':
                self.setStream('around_look.mp4')
            elif config.emitter == 'love':
                self.setStream('love_face.mp4')
            elif config.emitter == 'mindfulness':
                self.setStream('mindfulness.mp4')
            elif config.emitter == 'news':
                self.setStream('news.mp4')
            elif config.emitter == 'weather':
                self.setStream('weather.mp4')
            elif config.emitter == 'reset':
                self.setStream('happy_face.mp4', loop=False) # set True for continuesly moving eyes
                config.emitter = config.preEm = ''

        config.preEm = config.emitter

    def setStream(self, mp4, loop=False):
        if loop:
            self.loop = loop
            self.mp4 = mp4
        try:
            self.vs = cv2.VideoCapture(self.prefix + self.mp4)
        except Exception:
            self.vs = cv2.VideoCapture(self.prefix + mp4)

        if self.end:
            self.playVideo()

    def playVideo(self):
        self.sizes = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ret, frame = self.vs.read()

        if ret == False:
            self.end = True
            if self.loop:
                self.setStream(self.mp4, self.loop)
            else:
                config.emitter = config.preEm = ''
                return True

        if ret:
            frame = cv2.resize(frame, (self.sizes[0],self.sizes[1]))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

        self.root.after(1, self.playVideo)

    def destructor(self):
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


def mainframe():
    try:
        header('Running Eleonora version '+ config.VERSION)

        # start the app
        pba = Application()

        # Start Main Engine
        face_thread = threading.Thread(name='Face', target=Engine.engine)
        face_thread.setDaemon(True)
        face_thread.start()

        pba.root.mainloop()

    except KeyboardInterrupt:
        print (config.R + '\n' + config.O + '  • interrupted by user\n' + config.W)

    except EOFError:
        print (config.R + '\n' + config.O + '  • interrupted by error\n' + config.W)

if __name__ == '__main__':
    mainframe()
