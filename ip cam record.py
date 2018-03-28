
import cv2
import urllib2
import numpy as np
import sys
import keyboard
import pickle

def checkKeys():
	checkarray = [0,0,0,0]
	if (keyboard.is_pressed('up')):
		checkarray[0] = 1
	if (keyboard.is_pressed('down')):
		checkarray[1] = 1
	if (keyboard.is_pressed('left')):
		checkarray[2] = 1
	if (keyboard.is_pressed('right')):
		checkarray[3] = 1
	return checkarray


# Stream Video with OpenCV from an Android running IP Webcam (https://play.google.com/store/apps/details?id=com.pas.webcam)
# Code Adopted from http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera

host = "192.168.1.3:8080"
if len(sys.argv)>1:
	host = sys.argv[1]

hoststr = 'http://' + host + '/video'
print 'Streaming ' + hoststr

stream=urllib2.urlopen(hoststr)

bytes=''


try:
	RCdatabase = pickle.load( open("Database/TrainingData.p", "rb" ))
except:
	RCdatabase = []

while True:
	bytes+=stream.read(1024)
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')
	if a!=-1 and b!=-1:
		jpg = bytes[a:b+2]
		bytes= bytes[b+2:]
		i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
		# cv2.imshow("camvideo",i)
		chk = checkKeys()
		if (chk != [0,0,0,0]):
			RCdatabase.append([i,chk])
			print(chk)
		if (keyboard.is_pressed('e')):
			break

pickle.dump(RCdatabase, open("Database/TrainingData.p", "wb"))
