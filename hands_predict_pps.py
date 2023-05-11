#!/usr/bin/env python3

from predict_pps import dispatch_prediction

from io import StringIO

# import asyncio
from dataclasses import dataclass

from multiprocessing.dummy import Pool


import requests
import pandas as pd
import numpy as np

from itertools import chain

import cv2
import mediapipe as mp
import subprocess


pool = Pool(10)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(1)
data = []

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 5
fontColor = (100, 255, 255)
thickness = 2
lineType = 2


@dataclass
class Label:
    label: str


l = Label("")
l_prev = Label("")


def main():
    global l
    global l_prev
    i = 0
    while True:
        i += 1
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            ret, image = cam.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            cv2.putText(
                image,
                l.label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

            cv2.imshow("img", image)

            keypress = cv2.waitKey(1) & 0xFF
            n_frames = 5
            if keypress == ord("q"):
                break
            elif results.multi_hand_landmarks and i % n_frames == 0:
                landmarks = results.multi_hand_landmarks[0].landmark
                flat = chain.from_iterable([[nl.x, nl.y, nl.z] for nl in landmarks])
                f2 = {i: v for i, v in enumerate(flat)}
                pred = dispatch_prediction([f2])
                print(pred)
                l.label = pred
            elif i % n_frames == 0:
                l.label = ""

            l_prev = Label(l.label)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
