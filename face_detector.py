import cv2
import numpy as np
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread('images/eye_face.jpg')
fix_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)

faces = face_classifier.detectMultiScale(image, 1.3,5)

if len(faces) == 0:
	print('No faces detected')


def detect_face(fix_img):
	face_rects = face_classifier.detectMultiScale(fix_img)
	color = np.random.randint(0, 255, size=(3,))
	for(x,y,w,h) in face_rects:
		cv2.rectangle(fix_img,(x+y),(x+w,y+h),tuple(color),10)

	return fix_img


result = detect_face(fix_img)
plt.imshow(result)