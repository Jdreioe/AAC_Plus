# Importing the Libraries Required

import time
import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
from string import ascii_lowercase
from random import *

detector = HandDetector(detectionCon=0.7, maxHands=2)

# Creating and Collecting testing Data

mode = 'trainingData'
directory = 'dataSet/' + mode + '/'
minValue = 35

capture = cv2.VideoCapture(0)

interrupt = -1

def train( x ):
	TIMER = 3
	prev = time.time()
	while TIMER >= 0:
		_, frame = capture.read()

		# Simulating mirror Image

		frame = cv2.flip(frame, 1)

		# Getting count of existing images

		# Printing the count of each set on the screen

		cv2.putText(frame, "0 : " +str(count['zero']), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "a : " +str(count['a']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "b : " +str(count['b']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "c : " +str(count['c']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "d : " +str(count['d']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "e : " +str(count['e']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "f : " +str(count['f']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "g : " +str(count['g']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "h : " +str(count['h']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "i : " +str(count['i']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "k : " +str(count['k']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "l : " +str(count['l']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "m : " +str(count['m']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "n : " +str(count['n']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "o : " +str(count['o']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "p : " +str(count['p']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "q : " +str(count['q']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "r : " +str(count['r']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "s : " +str(count['s']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "t : " +str(count['t']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "u : " +str(count['u']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "v : " +str(count['v']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "w : " +str(count['w']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "x : " +str(count['x']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "y : " +str(count['y']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "z : " +str(count['z']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, "æ : " +str(count['æ']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
		cv2.putText(frame, str(TIMER),
				(200, 250), cv2.FONT_HERSHEY_PLAIN,
				7, (0, 255, 255),
				4, cv2.LINE_AA)


		hands = detector.findHands(frame, draw=False)  # with draw

		if len(hands)==1 and str(hands[0]["type"]) == "Right":
			bbox1 = hands[0]["bbox"] # x, y, w, h
			# To prevent a situation where roi x1 and roi y1 < 1, if bbox1 < 51, set x1 & y1 to 51
			if int(bbox1[0]) < 51:
				x1 = 51
			else:
				x1 = int(bbox1[0])

			x2 = int(bbox1[2]+x1)
			if int(bbox1[1]) < 51:
				y1 = 51
			else:
				y1 = int(bbox1[1])
			y2 = int(bbox1[3]+y1)


			
			

		else:

			x1 = int(0.5*frame.shape[1])
			y1 = 50
			x2 = frame.shape[1]-10
			y2 = int(0.5*frame.shape[1])

			# Drawing the ROI
			# The increment/decrement by 51 is to compensate for the bounding box

		cv2.rectangle(frame, (x1-51, y1-51), (x2+51, y2+51), (255,0,0) ,1)

			# Extracting the ROI

		roi = frame[y1-50:y2+50, x1-50:x2+50]

		cv2.imshow("Frame", frame)

		# Image Processing

		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),2)

		th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
		ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		# Output Image after the Image Processing that is used for data collection

		test_image = cv2.resize(test_image, (300,300))
		cv2.imshow("test", test_image)
		cur = time.time()
		interrupt = cv2.waitKey(10)
		if cur-prev >= 1:
			prev = cur
			TIMER = TIMER-1
		

	else:
		if str(x) == "0":
			cv2.imwrite(directory+"0/"+str(count['zero']+1)+'.jpg', test_image)
		else: 
			cv2.imwrite(directory+x.capitalize() + "/" +str(count[x]+1)+'.jpg', test_image)

while True:
	_, frame = capture.read()

	# Simulating mirror Image
	frame = cv2.flip(frame, 1)

	# Getting count of existing images

	count = {
				'zero': len(os.listdir(directory+"/0")),

				'a': len(os.listdir(directory+"/A")),
				'b': len(os.listdir(directory+"/B")),
				'c': len(os.listdir(directory+"/C")),
				'd': len(os.listdir(directory+"/D")),
				'e': len(os.listdir(directory+"/E")),
				'f': len(os.listdir(directory+"/F")),
				'g': len(os.listdir(directory+"/G")),
				'h': len(os.listdir(directory+"/H")),
				'i': len(os.listdir(directory+"/I")),
				'j': len(os.listdir(directory+"/J")),
				'k': len(os.listdir(directory+"/K")),
				'l': len(os.listdir(directory+"/L")),
				'm': len(os.listdir(directory+"/M")),
				'n': len(os.listdir(directory+"/N")),
				'o': len(os.listdir(directory+"/O")),
				'p': len(os.listdir(directory+"/P")),
				'q': len(os.listdir(directory+"/Q")),
				'r': len(os.listdir(directory+"/R")),
				's': len(os.listdir(directory+"/S")),
				't': len(os.listdir(directory+"/T")),
				'u': len(os.listdir(directory+"/U")),
				'v': len(os.listdir(directory+"/V")),
				'w': len(os.listdir(directory+"/W")),
				'x': len(os.listdir(directory+"/X")),
				'y': len(os.listdir(directory+"/Y")),
				'z': len(os.listdir(directory+"/Z")),
				'æ': len(os.listdir(directory+"/Æ")),
	}


	# Printing the count of each set on the screen
	
	cv2.putText(frame, "0 : " +str(count['zero']), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "a : " +str(count['a']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "b : " +str(count['b']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "c : " +str(count['c']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "d : " +str(count['d']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "e : " +str(count['e']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "f : " +str(count['f']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "g : " +str(count['g']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "h : " +str(count['h']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "i : " +str(count['i']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "k : " +str(count['k']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "l : " +str(count['l']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "m : " +str(count['m']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "n : " +str(count['n']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "o : " +str(count['o']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "p : " +str(count['p']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "q : " +str(count['q']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "r : " +str(count['r']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "s : " +str(count['s']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "t : " +str(count['t']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "u : " +str(count['u']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "v : " +str(count['v']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "w : " +str(count['w']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "x : " +str(count['x']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "y : " +str(count['y']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "z : " +str(count['z']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
	cv2.putText(frame, "æ : " +str(count['æ']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


	# Coordinates of the ROI
	
	hands = detector.findHands(frame, draw=False)  # with draw
	if (len(hands)==1 and str(hands[0]["type"]) == "Left"):

		bbox1 = hands[0]["bbox"] # x, y, w, h
		if int(bbox1[0]) < 51:
			x1 = 51
		else:
			x1 = int(bbox1[0])

		x2 = int(bbox1[2]+x1)
		if int(bbox1[1]) < 51:
			y1 = 51
		else:
			y1 = int(bbox1[1])
		y2 = int(bbox1[3]+y1)


		
		

	else:

		x1 = int(0.5*frame.shape[1])
		y1 = 50
		x2 = frame.shape[1]-10
		y2 = int(0.5*frame.shape[1])

		# Drawing the ROI
		# The increment/decrement by 51 is to compensate for the bounding box

	cv2.rectangle(frame, (x1-51, y1-51), (x2+51, y2+51), (255,0,0) ,1)

		# Extracting the ROI

	roi = frame[y1-50:y2+50, x1-50:x2+50]

	cv2.imshow("Frame", frame)
	
	# Image Processing

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),2)
		
	th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	
	# Output Image after the Image Processing that is used for data collection 

	test_image = cv2.resize(test_image, (300,300))
	cv2.imshow("test", test_image)

	# Data Collection

	interrupt = cv2.waitKey(10)
	if interrupt & 0xFF == 27: 
		# esc key
		break
	# rest of if-statements are basicly the same: if key == ord('x') is pressed, do train('x')
	if interrupt & 0xFF == ord('0'):
		
		train("0")
	elif interrupt & 0xFF == ord('a'):
		train("a")

	elif interrupt & 0xFF == ord('b'):
		train("b")
	
	elif interrupt & 0xFF == ord('c'):
		train("c")
	
	elif interrupt & 0xFF == ord('d'):
		train("d")
	
	elif interrupt & 0xFF == ord('e'):
		train("e")
	
	elif interrupt & 0xFF == ord('f'):
		train("f")
	
	elif interrupt & 0xFF == ord('g'):
		train("g")
	
	elif interrupt & 0xFF == ord('h'):
		train("h")
	
	elif interrupt & 0xFF == ord('i'):
		train("i")
	
	elif interrupt & 0xFF == ord('j'):
		train("j")
	
	elif interrupt & 0xFF == ord('k'):
		train("k")
	
	elif interrupt & 0xFF == ord('l'):
		train("l")
	
	elif interrupt & 0xFF == ord('m'):
		train("m")
	
	elif interrupt & 0xFF == ord('n'):
		train("n")
	
	elif interrupt & 0xFF == ord('o'):
		train("o")
	
	elif interrupt & 0xFF == ord('p'):
		train("p")
	
	elif interrupt & 0xFF == ord('q'):
		train("q")
	
	elif interrupt & 0xFF == ord('r'):
		train("r")
	
	elif interrupt & 0xFF == ord('s'):
		train("s")
	
	elif interrupt & 0xFF == ord('t'):
		train("t")
	
	elif interrupt & 0xFF == ord('u'):
		train("u")
	
	elif interrupt & 0xFF == ord('v'):
		train("v")
	
	elif interrupt & 0xFF == ord('w'):
		train("w")
	
	elif interrupt & 0xFF == ord('x'):
		train("x")
	
	elif interrupt & 0xFF == ord('y'):
		train("y")
	
	elif interrupt & 0xFF == ord('z'):
		train("z")
	elif interrupt & 0xFF == ord('æ'):
		train("æ")
	if interrupt != 0xff:
		cv2.putText(frame, str("Not "),
				(200, 250), cv2.FONT_HERSHEY_PLAIN,
				7, (0, 255, 255),
				4, cv2.LINE_AA)
	if interrupt & 0xff == ord('.'):
		i = int(input("Select the number of letters that you would like to generate"))
		
		d = []
		for t in range(len(ascii_lowercase)):
			d.append(ascii_lowercase[t])

		d.append('æ')
		while i > 0:

			letter = d[randint(0, 26)]
			
			cv2.putText(frame, str(letter),
				(300, 300), cv2.FONT_HERSHEY_PLAIN,
				7, (0, 255, 255),
				4, cv2.LINE_AA)
			print(letter)
			train(str(letter))
			i-=1

		print("Done")




	
capture.release()
cv2.destroyAllWindows()
