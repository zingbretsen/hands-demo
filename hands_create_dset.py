#!/usr/bin/env python3

from itertools import chain
import uuid

import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(1)
data = []

while True:
    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
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
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 25))

        keypress = cv2.waitKey(1) & 0xFF
        tag = None
        if keypress == ord("q"):
            break
        elif keypress == ord("l"):
            tag = "love"
        elif keypress == ord("h"):
            tag = "horns"
        elif keypress == ord("o"):
            tag = "OK"
        elif keypress == ord("p"):
            tag = "peace"
        elif keypress == ord("t"):
            tag = "thumbsup"
        elif keypress == ord("s"):
            tag = "spock"
        elif keypress == ord("n"):
            tag = "none"
        if tag:
            print(len(results.multi_hand_landmarks))
            fname = f"imgs/{tag}/{uuid.uuid4()}.png"
            cv2.imwrite(fname, image)
            landmarks = results.multi_hand_landmarks[0].landmark
            flat = chain.from_iterable([[nl.x, nl.y, nl.z] for nl in landmarks])
            f2 = {i: v for i, v in enumerate(flat)}
            f2["target"] = tag
            f2["fname"] = fname
            data.append(f2)

cam.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)

uid = str(uuid.uuid4())
df.to_csv(f"./hands_{uid}.csv", index=False)
