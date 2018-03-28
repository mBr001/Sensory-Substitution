from PIL import Image
from PIL import ImageGrab
import numpy as np
import cv2

import keyboard


img = ImageGrab.grab(bbox=(100,10,400,780))
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
cv2.imwrite('img.png', img)

# im=ImageGrab.grab(bbox=(10,10,500,500))
# im.save('im.png')
while True:
	if (keyboard.is_pressed('down')):
		print("pressed")