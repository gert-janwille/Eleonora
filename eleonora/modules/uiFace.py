import cv2
import time
import numpy as np
from AppKit import NSScreen

class CreateFace(object):
    """
    Creates a canvas and draw Eleonora's face.
    Other things in the class are in function
    of the face of Eleonora
    """
    def __init__(self, name, bgcolor, eyeWidth, eyeHeight, eyeSpace, eyeColor):
        # Get screen sizes
        screenWidth, screenHeight = _getScreenSizes()

        # Variables
        self.name = name
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.bgcolor = bgcolor
        self.eyeWidth = eyeWidth
        self.eyeHeight = eyeHeight
        self.eyeSpace = int((screenWidth / 2) / eyeSpace)
        self.eyeColor = eyeColor

        # Create new full screen window
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Make new image with color
        self.canvas = np.zeros((self.screenHeight, self.screenWidth, 3), np.uint8)
        self.canvas.fill(self.bgcolor)

        # Initialize Eyes
        self.eyes(0, 0)

    def sleep(self, dx, dy):
        self.eyes(dx, dy,100, "closed")

    def moveEyes(self, dx, dy, closing = 0):
        self.eyes(dx, dy, closing)

    def closedEye(self, pos):
        cv2.ellipse(self.canvas,(pos[0],pos[1]),(self.eyeWidth,self.eyeHeight),0,0,180,0,-1)
        cv2.ellipse(self.canvas,(pos[0],pos[1]-20),(self.eyeWidth,self.eyeHeight),0,0,180,(255,255,255),-1)

    def openEye(self, pos, size):
            cv2.ellipse(self.canvas, pos, size, 0, 0, 360, (0,0,0), -1)

    def eyes(self, dx, dy, closing=0, eyeType="open"):
        # Get center of canvas
        centerWidth, centerHeight = int(self.screenWidth / 2), int(self.screenHeight / 2)

        # Calculate the left and right eye's position
        leftEye = (int(centerWidth - self.eyeSpace + dx - ((self.eyeWidth + (self.eyeWidth/2)) * 2)), int(centerHeight + dy - self.eyeHeight))
        rightEye = (int(centerWidth + self.eyeSpace + dx - ((self.eyeWidth + (self.eyeWidth/2)) * 2)), int(centerHeight + dy - self.eyeHeight))

        # Clear the canvas
        self.draw()

        # Draw the eyes
        if eyeType == "closed":
            self.closedEye((int(centerWidth - self.eyeSpace),centerHeight))
            self.closedEye((int(centerWidth + self.eyeSpace),centerHeight))

        if eyeType == "open":
            self.openEye(leftEye, (int(self.eyeWidth), int(self.eyeHeight - closing)))
            self.openEye(rightEye, (int(self.eyeWidth), int(self.eyeHeight - closing)))

    def draw(self):
        self.canvas.fill(self.bgcolor)

    def type(self, pos=(0,0), string="Hello G", color=(0,0,0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.canvas, string,pos, font, 1, color, 2)

    def show(self):
        cv2.imshow(self.name, self.canvas)

def _getScreenSizes():
    width = int(NSScreen.mainScreen().frame().size.width)
    height = int(NSScreen.mainScreen().frame().size.height)
    return width, height
