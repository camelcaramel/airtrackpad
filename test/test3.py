from cv2 import cv2
import math

face_cascade = cv2.CascadeClassifier(
    'airtrackpad\data\haarcascade_frontalface_default.xml')

# 트랙커 객체 생성자 함수 리스트 ---①
trackers = [cv2.TrackerBoosting_create,
            cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerTLD_create,
            cv2.TrackerMedianFlow_create,
            cv2.TrackerCSRT_create,
            cv2.TrackerMOSSE_create]

trackerIdx = 2  # 트랙커 생성자 함수 선택 인덱스s
tracker = trackers[trackerIdx]()

cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
# delay = int(1000/fps)
win_name = 'test3'

# status var
# 0 : nothing 1 : find face 2 : lost
status = 0

cur_position = None


postions = {"start": None, "end": None}


lost_target_count = 0

th_value = 5
length_th_value = 10


faces = None

fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
delay = int(1000/fps)


def init_var() -> None:
    global lost_target_count
    global status
    global postions
    lost_target_count = 0
    status = 0
    postions['start'] = None
    postions['end'] = None
    print("initailize")


def action1(img) -> None:
    """action stub 1"""
    cv2.putText(img, "action1",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    init_var()
    print("action1 activated")


def action2(img) -> None:
    """action stub 2"""
    cv2.putText(img, "action2",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    init_var()
    print("action2 activated")


def action3(img) -> None:
    """action stub 3"""
    cv2.putText(img, "action3",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    init_var()
    print("action3 activated")


def action4(img) -> None:
    """action stub 4"""
    cv2.putText(img, "action4",
                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    init_var()
    print("action4 activated")


def branch(start: list, end: list, img):
    print("on branch")
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]

    abs_gradient = abs(end_y-start_y)/(end_x-start_x)
    length = length = math.sqrt((end_x-start_x)**2+(end_y-start_y)**2)
    vec = [end_x-start_x, end_y-start_y]

    if length > length_th_value:
        if vec[0] > 0 and vec[1] > 0:
            if abs_gradient >= 1:
                action1(img)
            else:
                action2(img)
        elif vec[0] > 0 and vec[1] < 0:
            if abs_gradient >= 1:
                action3(img)
            else:
                action2(img)
        elif vec[0] < 0 and vec[1] > 0:
            if abs_gradient >= 1:
                action3(img)
            else:
                action4(img)
        else:
            if abs_gradient >= 1:
                action1(img)
            else:
                action4(img)
    else:
        init_var()
        print("too short")


frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    img_draw = frame.copy()

    if ret:
        if status == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                                  minNeighbors=5, minSize=(80, 80))
            if len(faces) != 0:
                isInit = tracker.init(frame, tuple(faces[0]))
                if isInit:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    print("cascde find face")
                    postions['start'] = [faces[0][0], faces[0][1]]
                    status = 1
        elif status == 1:
            ok, bbox = tracker.update(frame)
            print(f"status1  {ok} :   {bbox}")
            (x, y, w, h) = bbox
            cur_position = [x, y]
            if ok:
                cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 255, 0), 2, 1)
            else:
                status = 2
        else:
            ok, bbox = tracker.update(frame)
            if ok:
                status = 1
                (x, y, w, h) = bbox
                cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 255, 0), 2, 1)
                cur_position = [x, y]
            else:
                if lost_target_count > th_value and postions['start']:
                    postions['end'] = cur_position
                    branch(postions['start'], postions['end'], img_draw)
                    status = 0
                lost_target_count += 1
        frame_count += 1

    key = cv2.waitKey(delay) & 0xff

    print(f"{frame_count} :  {status}")
    cv2.imshow(win_name, img_draw)


cap.release()
cv2.destroyAllWindows()
