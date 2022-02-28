# Importing the Libraries Required

import time
import cv2
import numpy as np
import os

# Creating and Collecting testing Data

mode = 'testingData'
directory = 'dataSet/' + mode + '/'
minValue = 35

capture = cv2.VideoCapture(0)
interrupt = -1

def test( x ):
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


		# Coordinates of the ROI

		x1 = int(0.5*frame.shape[1])
		y1 = 10
		x2 = frame.shape[1]-10
		y2 = int(0.5*frame.shape[1])

		# Drawing the ROI
		# The increment/decrement by 1 is to compensate for the bounding box

		cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)

		# Extracting the ROI

		roi = frame[y1:y2, x1:x2]

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
		print(x)
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
	
	x1 = int(0.5*frame.shape[1])
	y1 = 10
	x2 = frame.shape[1]-10
	y2 = int(0.5*frame.shape[1])

	# Drawing the ROI
	# The increment/decrement by 1 is to compensate for the bounding box
	
	cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
	
	# Extracting the ROI
	
	roi = frame[y1:y2, x1:x2]

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
	# rest of if-statements are basicly the same: if key == ord('x') is pressed, do test('x') 
	if interrupt & 0xFF == ord('0'):
		
		test("0")
	elif interrupt & 0xFF == ord('a'):
		test("a")

	elif interrupt & 0xFF == ord('b'):
		test("b")
	
	elif interrupt & 0xFF == ord('c'):
		test("c")
	
	elif interrupt & 0xFF == ord('d'):
		test("d")
	
	elif interrupt & 0xFF == ord('e'):
		test("e")
	
	elif interrupt & 0xFF == ord('f'):
		test("f")
	
	elif interrupt & 0xFF == ord('g'):
		test("g")
	
	elif interrupt & 0xFF == ord('h'):
		test("h")
	
	elif interrupt & 0xFF == ord('i'):
		test("i")
	
	elif interrupt & 0xFF == ord('j'):
		test("j")
	
	elif interrupt & 0xFF == ord('k'):
		test("k")
	
	elif interrupt & 0xFF == ord('l'):
		test("l")
	
	elif interrupt & 0xFF == ord('m'):
		test("m")
	
	elif interrupt & 0xFF == ord('n'):
		test("n")
	
	elif interrupt & 0xFF == ord('o'):
		test("o")
	
	elif interrupt & 0xFF == ord('p'):
		test("p")
	
	elif interrupt & 0xFF == ord('q'):
		test("q")
	
	elif interrupt & 0xFF == ord('r'):
		test("r")
	
	elif interrupt & 0xFF == ord('s'):
		test("s")
	
	elif interrupt & 0xFF == ord('t'):
		test("t")
	
	elif interrupt & 0xFF == ord('u'):
		test("u")
	
	elif interrupt & 0xFF == ord('v'):
		test("v")
	
	elif interrupt & 0xFF == ord('w'):
		test("w")
	
	elif interrupt & 0xFF == ord('x'):
		test("x")
	
	elif interrupt & 0xFF == ord('y'):
		test("y")
	
	elif interrupt & 0xFF == ord('z'):
		test("z")
	elif interrupt & 0xFF == ord('æ'):
		test("æ")
	if interrupt != 0xff: 
		cv2.putText(frame, str("Not "),
				(200, 250), cv2.FONT_HERSHEY_PLAIN,
				7, (0, 255, 255),
				4, cv2.LINE_AA)



	
capture.release()
cv2.destroyAllWindows()
