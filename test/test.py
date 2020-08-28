from cv2 import cv2
import numpy as np
import math

# 얼굴과  검출을 위한 케스케이드 분류기 생성
face_cascade = cv2.CascadeClassifier(
    'data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)

# coordinate-related values
current_postion = np.zeros((1, 4))
points = {"start": None, "end": None}

status = 0
# 0 : wait / 1 : detect / 2: lost


# frame count vars
frame_count = 0
no_face_frame_count = 0


# threshhold value of change status
status_change_th_frame = 10


def init_var() -> None:
    global no_face_frame_count
    global status
    global points
    no_face_frame_count = 0
    status = 0
    points['start'] = None
    points['end'] = None
    print("initailize")


def action1() -> None:
    """action stub 1"""
    init_var()
    print("action1 activated")


def action2():
    """action stub 2"""
    init_var()
    print("action2 activated")


def cal(start: list, end: list) -> int:
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]

    gradient = ((start_x)-(end_y))/((start_y)-(end_x))
    length = math.sqrt((end_x-start_x)**2+(end_y-start_y)**2)

    if length > 10:
        if gradient > 0 and gradient < 4:
            action1()
        elif gradient < 0 and gradient > -4:
            action2()
        else:
            init_var()
            print("finished in level2")
    else:
        init_var()
        print("too short")


while cap.isOpened():
    ret, img = cap.read()  # 프레임 읽기

    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                              minNeighbors=5, minSize=(80, 80))
        if cv2.waitKey(5) == 27:
            break

        frame_count += 1
        print(f"{frame_count}  :  {faces}  :  {len(faces)} : {points}")

        if len(faces) == 0:
            if status == "0":
                continue
            elif status == 1:
                status = 2
                no_face_frame_count += 1
            else:
                if no_face_frame_count > status_change_th_frame and points['start']:
                    points["end"] = current_postion
                    status = 0
                    cal(points['start'], points['end'])
                    continue
                no_face_frame_count += 1
        else:
            if status == 0:
                status = 1
                current_postion = (faces[0][0], faces[0][1])
                points['start'] = current_postion
            elif status == 1:
                current_postion = (faces[0][0], faces[0][1])
            else:
                no_face_frame_count = 0
                status = 1

        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]

        cv2.imshow('face detect', img)
    else:
        break

cv2.destroyAllWindows()
