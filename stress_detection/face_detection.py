import cv2
import dlib
import numpy as np

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
    for rect in rects:
        shape = predictor(gray_img, rect)
        # convert to np array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np.copy()

        # display the landmarks
        
        for i, (x, y) in enumerate(shape):
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # display the image 
    cv2.imshow("Landmark Detection", image)
    

    # press Esc to terminate the session
    if cv2.waitKey(10) == 27:
        break


cap.release()
        
