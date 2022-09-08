import os
import cv2



def image_creator(path_to_directory, label):
    """This function takes and saves image"""
    cap = cv2.VideoCapture(0)
    num_of_image = len(os.listdir(path_to_directory))
    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        cv2.imwhow("frame", rgb)
        if cv2.waitkey(1) & 0xFF == ord("q"):
            out = cv2.imwrite(os.path.join(path_to_directory, f"{label}_image{num_of_image}.jpg"), frame)
            break
    return None