from grabscreen import grab_screen
import cv2

screen = grab_screen(region=(0, 40, 800, 640))
cv2.imshow('image', screen)
cv2.waitKey(0)

