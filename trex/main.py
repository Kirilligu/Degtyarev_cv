import numpy as np
import pyautogui as pg
import mss
import keyboard

i = 1
w = 82
monitor = {'top':320, 'left':78, 'width': w, 'height': 25}
restart_coords = (300,320)
sct = mss.mss()
pg.click(restart_coords)
while True:
    img = np.array(sct.grab(monitor))

    if img.mean()<249:
        pg.press('space')

    if w < 345:
        i +=1
        if i == 100:
            w+=3
            monitor = {'top':320, 'left':78, 'width': w, 'height': 25}
            i = 1

    if keyboard.is_pressed('q'):
        print('breaking')
        break
