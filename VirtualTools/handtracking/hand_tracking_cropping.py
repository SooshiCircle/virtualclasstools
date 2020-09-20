import cv2
from utils import detector_utils


ERROR_THRESH = 200
SCORE_THRESH = 0.08
CC_BOOL = False
COORDS_BOOL = True
PAINT_BOOL = True
STROKE_BOOL = True
STROKES = []
CENTERS = []
GOODNESS = 25
RAISED_THRESH = 200
CHECK_LEN = 30

def dist(l1, l2):
    return ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)**(1/2)

def fltr(listy, coordtoAdd):
    if listy != []:
        return dist(listy[-1], coordtoAdd) < ERROR_THRESH
    else:
        return True

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def hand_raised(coords):
    if len(coords) >= CHECK_LEN:
        # print(coords)
        l = coords[-CHECK_LEN: len(coords)]
        # print(l)
        x = [i[0] for i in l]
        y = [i[1] for i in l]

        avgx = sum(x)/len(x)
        avgy = sum(y)/len(y)

        good = 0
        for i in range(CHECK_LEN):
            if abs(x[i] - avgx) < RAISED_THRESH and abs(y[i] - avgy) < RAISED_THRESH:
                good += 1
        if good > GOODNESS:
            return True
        else:
            return False

detection_graph, sess = detector_utils.load_inference_graph()

if CC_BOOL:
    CROP_COUNTER = [0]
else:
    CROP_COUNTER = None

if PAINT_BOOL:
    paintCoords = []
else:
    paintCoords = None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    im_width, im_height = (cap.get(3), cap.get(4))
    
    num_hands_detect = 1 # max number of hands to detect

    while True:
        ret, frame = cap.read()
#         frame = rescale_frame(frame, 30)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

        COORDS = []
        detector_utils.draw_box_on_image(num_hands_detect, SCORE_THRESH,
                                         scores, boxes, im_width, im_height,
                                         frame, CROP_COUNTER, COORDS, CENTERS)
        if hand_raised(CENTERS):
            # print("CENTERS: {}".format(len(CENTERS)))
            print("HAND IS RAISED")

        # if paintCoords == [] and STROKES == []:
            # print("AHAAA")
        if STROKE_BOOL:
            if paintCoords != None:
                for i in COORDS:
                    if fltr(paintCoords, i):
                        paintCoords.append(i)
        for point in range(len(paintCoords[:-2])):
            cv2.line(frame, pt1=(paintCoords[point][0], paintCoords[point][1]), pt2=(paintCoords[point + 1][0], paintCoords[point + 1][1]), color=(0, 0, 255), thickness=8)
        for stroke in STROKES:
            for point in range(len(stroke[:-2])):
                cv2.line(frame, pt1=(stroke[point][0], stroke[point][1]), pt2=(stroke[point + 1][0], stroke[point + 1][1]), color=(0, 0, 255), thickness=8)

        cv2.imshow('hand_tracking', cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1))
        if cv2.waitKey(25) & 0xFF == ord('s'):
            if STROKE_BOOL:
                STROKE_BOOL = False
                STROKES.append(paintCoords.copy())
                paintCoords = []
                # print("here s T")
            else:
                STROKE_BOOL = True
                # print("here s F")
        if cv2.waitKey(25) & 0xFF == ord('c'):
            paintCoords = []
            STROKES = []
            # print("here c")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
