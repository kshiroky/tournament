from multiprocessing.connection import answer_challenge
import cv2
import dlib
import numpy as np
import pandas as pd
import imutils
from imutils import face_utils
from scipy.spatial import distance

def eyebrow_dist(left_eb, right_eb):
    dist = distance.euclidean(left_eb, right_eb)
    eyebrow_points.append(dist)
    return dist


def lip_dist(top_point, bott_point):
    dist = distance.euclidean(top_point, bott_point)
    mouth_points.append(dist)
    return dist
    

def min_max_stress_scaler(dist_eb, dist_mouth):
    # quiz_result = process_quiz_answers(answer_list=answer_list)
    norm_eb_dist = abs(dist_eb) 
    log_eb_dist = np.log10(norm_eb_dist)
    norm_mouth_dist = abs(dist_mouth) 
    log_mouth_dist = np.log10(norm_mouth_dist)
    pre_stress_value = (log_eb_dist + log_mouth_dist) / 2
    stress_value = np.exp(-(pre_stress_value))
    if stress_value > 0.5:
        stress_label = "Stressed"
    else:
        stress_label = "Not stressed"
    return stress_label, stress_value


def process_quiz_answers(answer_list):
    num_answ = len(answer_list)
    norm_answer_value = 0
    for answer in answer_list:
        norm_value = int(answer) / num_answ
        norm_answer_value += norm_value
    return norm_answer_value

#  
answer_list = []

question_1 = int(input("Сколько часов вы сегодня спали? \n Введите число от 0 до 12: \n"))
if int(question_1) <= 9: 
    answer_1 = question_1 / 9 
if question_1 > 9:
    answer_1 = question_1 / 9 - (question_1 - 9) / 9
answer_list.append(answer_1)

question_2 = int(input("Оцените свое самочувствие по шкале от 1 до 5: \n"))
answer_2 = question_2 / 5
answer_list.append(answer_2)


print("Оцените свое самочувствие:")
question_3 = input("Были ли у вас сегодня конфликты или споры (да/нет)?\n")
if "да" in question_3:
    answer_3 = 1
elif "нет" in question_3:
    answer_3 = 0
answer_list.append(answer_3)

question_4 = int(input("Как часто вы дотрагиваетесь до своего лица (от 0 до 7), \n где 0 - совсем не дотрагивались, 7 - очень часто касаетесь:\n "))
answer_4 = question_4 / 7
answer_list.append(answer_4)


question_5 = int(input("Сколько чашек кофе вы сегодня выпили?"))
if question_5 == 0 is False:
    answer_5 = question_5 / max(question_5)
answer_list.append(answer_5)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)


while True:
    # capture the image from the webcam
    ret, image = cap.read()
  
    # convert the image color to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    # detect the face 
    rects = detector(gray_img, 1)

    # detect landmarks for face
    (l_begin, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (r_begin, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (lips_lower, lips_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    for rect in rects:
        shape = predictor(gray_img, rect)
        # convert to np array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np.copy()

        # detect the parts of face
        left_eb = shape[l_begin:l_end]
        right_eb = shape[r_begin:l_begin]
        mouth = shape[lips_lower:lips_upper]
        
        # make hull transform of landmarks
        left_eb_hull = cv2.convexHull(left_eb)
        right_eb_hull = cv2.convexHull(right_eb)
        mouth_hull = cv2.convexHull(mouth)
       

        # display the convexed landmarks
        cv2.drawContours(image, [left_eb_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [right_eb_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [mouth_hull], -1, (0, 255, 0), 1)
        
        # for i, (x, y) in enumerate(shape):
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        mouth_points = []
        eyebrow_points = []
        lip_distance = lip_dist(mouth_hull[-1], mouth_hull[0])
        eb_distance = eyebrow_dist(left_eb_hull[-1].flatten(), right_eb_hull[-1].flatten())

        # calculate stress-level
        label_stress, value_stress = min_max_stress_scaler(eb_distance, lip_distance)

        if pd.isna(value_stress * 100) == True:
            continue

        # picture text parameters
        text_one = f"Stress value: {str(round(value_stress * 100, 2))}"
        origin_one = (10, 40)
        text_two = f"Stress level: {label_stress}"
        origin_two = (10, 60)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text_one, origin_one, font, 0.5, (0, 0, 255), 2)
        cv2.putText(image, text_two, origin_two, font, 0.5, (0, 0, 255), 2)
        
    # # display the image zz
    cv2.imshow("Stress Level Detection", image)
    

    # press Esc to terminate the session
    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()
cap.release()
        
