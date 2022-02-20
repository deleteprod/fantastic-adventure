#!/usr/bin/env python3

'''
Code largely taken from the freeCodeCamp advanced computer vision thingy on YouTube
the one published on 27 May 2021 - https://www.youtube.com/watch?v=01sAkU_NvOY
'''

import cv2
import mediapipe as mp
import time

# Experimental - use numba for faster execution
# import numba

# Initiate the cam
capture = cv2.VideoCapture(0)

# Bring in what we need from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# For FPS
pTime = 0
cTime = 0

# Visualise
while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)

                # Height, width, and channel of the image
                height, weight, channel = img.shape

                # Convert decimal output of height/width to pixels
                cx, cy = int(lm.x*weight), int(lm.y*height)
                print(id, cx, cy)

                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 0), cv2.FILLED)

            # draw_landmark method overlays image from capture
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 240, 255), 3)

    cv2.imshow("Hand Landmarks", img)
    cv2.waitKey(1)

def main():
    pass


if __name__ == "__main__":
    main()