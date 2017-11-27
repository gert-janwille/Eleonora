from AppKit import NSScreen
import numpy as np
import cv2

width = int(NSScreen.mainScreen().frame().size.width)
height = int(NSScreen.mainScreen().frame().size.height)

cv2.namedWindow("canvas", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("canvas",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

canvas = np.zeros((height,width,3), np.uint8)
canvas.fill(255)

cv2.ellipse(canvas,(250,250),(80,100),0,0,180,0,-1)
cv2.ellipse(canvas,(250,230),(80,100),0,0,180,(255,255,255),-1)
#Display the image
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
