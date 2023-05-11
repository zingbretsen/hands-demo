#!/usr/bin/env python3
from io import StringIO

# import asyncio
from dataclasses import dataclass

from multiprocessing.dummy import Pool

pool = Pool(10)

import requests
import pandas as pd
import numpy as np

from itertools import chain

import cv2
import mediapipe as mp
import subprocess


def dispatch_webook(url):
    pool.apply_async(requests.get, [url])


red = "https://hooks.nabu.casa/gAAAAABiDlv9RyMDk4TirYylrEMgFnfWt0qqyMCeYSwqkneDm8kUf7NpnAhr4DWON79pviOnIBJDZfb-t48uVyg_h2SFgB0V4OrvZW3a0HxjhMwaLherYuW4gQLqXhbFmgFn5rlMwkdUIRq3CNK1PApwId0_yajI3feqS3olYc_Ahhl5w0N8TOQ="
green = "https://hooks.nabu.casa/gAAAAABiDlyh0dNJBGxKbOjlhtQzWjZnt5_JCKog5iNJoBLyPTVjbYqk1t1pGbJfszoZt3m1bG2ZdX_eTFzcgKHFtPfZxYjBJm0rsOVaOS5yqoplVpi8EMxvKS09ZoZT37ySRRyENfDy9wFedKxK4BxvfpM3Vk-BXP3FRx14cSJ7ipG0hNT_9Cg="
blue = "https://hooks.nabu.casa/gAAAAABiDlytufhxhvC2h7FakiFyjdE7LvQ5nk-_lYjB-Qv1-QXme_fzQocttRC46XX9g3dbu6cnV8YYb-blrDyMTD8S7_XneRa557z7rIwXUENHDAwQmEjUBQDAxsfL2o730DjxBWzn2KV_WJHlow_XsAOgIkHsFXO4W6fqbuA_avjQUWThtu0="
white = "https://hooks.nabu.casa/gAAAAABiDl58FfHR6zEdcCxgu08N02omrBKbKsL9JsecjJD2o2LgG6XMDeeFXk_Xq7R0ulRYg5v477QnOCZDQqlcsPS8KJG39R--Bs-I8thJR5Vpi-3e9kIrYzE8FYvNMaGGoYl2Dac6E8Dfp_iy8JRLNxCfMhCnOCmALDK1MYhvsfp1D4nx_7M="
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(2)
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


def dispatch_prediction(df, label=l, choices=["OK", "horns", "none", "peace", "spock", "thumbsup"]):
    df.to_csv("/tmp/to_predict.csv", index=False)
    out = subprocess.run(
        [
            "java",
            "-jar",
            # "/Users/zach.ingbretsen/old_downloads/63b45a070797080725e40522-63b44b69324b6d75284d9225.jar",
            "/Users/zach.ingbretsen/Documents/projects/hands/645bc81610a88465780f31ee-645bc4ea81a542c31b6413c2.jar",
            "csv",
            "--input=-",
            "--output=-",
        ],
        capture_output=True,
        input=bytes(df.to_csv(index=False), encoding="UTF-8"),
    )
    df = pd.read_csv(StringIO(out.stdout.decode()))
    label.label = df.apply(lambda row: choices[np.argmax(row)], axis=1).iat[0]
    # return df.apply(lambda row: choices[np.argmax(row)], axis=1).iat[0]


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
                # for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            n_frames = 10
            if results.multi_hand_landmarks and i % n_frames == 0:
                landmarks = results.multi_hand_landmarks[0].landmark
                flat = chain.from_iterable([[nl.x, nl.y, nl.z] for nl in landmarks])
                f2 = {i: v for i, v in enumerate(flat)}
                dispatch_prediction(pd.DataFrame([f2]))

            elif i % n_frames == 0:
                l.label = ""

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = hand.landmark
                height = 1080
                width = 1920
                x = list(chain.from_iterable([[nl.x] for nl in landmarks]))
                y = list(chain.from_iterable([[nl.y] for nl in landmarks]))
                z = list(chain.from_iterable([[nl.z] for nl in landmarks]))
                coords = (sum(x)/len(x), max(y), sum(z)/len(z))
                scaled_coords = [int(coords[0] * width), 30+int(coords[1] * height), coords[2]]
                print(coords)
                print(scaled_coords)
                print(scaled_coords[:2])

                cv2.putText(
                    image,
                    l.label,
                    scaled_coords[:2],
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                )
            print(l_prev.label, l.label)
            if l.label != l_prev.label:
                if l.label == "peace":
                    dispatch_webook(red)
                elif l.label == "thumbsup":
                    dispatch_webook(blue)
                elif l.label == "spock":
                    dispatch_webook(green)
                elif l.label == "none":
                    dispatch_webook(white)


            cv2.imshow("img", image)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
            l_prev = Label(l.label)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
