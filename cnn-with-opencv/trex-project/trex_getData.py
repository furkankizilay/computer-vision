import keyboard
import uuid # to record from the screen
import time
from PIL import Image
from mss import mss

"""
https://fivesjs.skipser.com/trex-game/
"""

mon = {"top":500, "left":680, "width":260, "height":125} 
sct = mss() # will cut and frame the relevant roi from the screen in line with the pixels

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print("{}: {}".format(key, i)) # key: button pressed from key keyboard, i: how many times the button is pressed
    img = sct.grab(mon) # get the screen
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("C:/Users/furka/Desktop/computer-vision/cnn-with-opencv/trex-project/img/{}_{}_{}.png".format(key, record_id, i))

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()

while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError: continue





