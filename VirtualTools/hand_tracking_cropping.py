from utils import detector_utils as detector_utils
import cv2

detection_graph, sess = detector_utils.load_inference_graph()

ERROR_THRESH = 200
SCORE_THRESH = 0.08
CC_BOOL = True
COORDS_BOOL = True
PAINT_BOOL = True

if CC_BOOL:
    CROP_COUNTER = [0]
else:
    CROP_COUNTER = None

COORDS = []

if PAINT_BOOL:
    paintCoords = []
else:
    paintCoords = None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    im_width, im_height = (cap.get(3), cap.get(4))
    
    num_hands_detect = 1 # max number of hands to detect

    while True:
        ret, frame = cap.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

        detector_utils.draw_box_on_image(num_hands_detect, SCORE_THRESH,
                                         scores, boxes, im_width, im_height,
                                         frame, CROP_COUNTER, COORDS)

        if paintCoords != None:
            for i in COORDS:
                if filter(paintCoords, i, ERROR_THRESH):
                    paintCoords.append(i)
            for point in range(len(paintCoords[:-2])):
                cv2.line(frame, pt1=(paintCoords[point][0], paintCoords[point][1]), pt2=(paintCoords[point + 1][0], paintCoords[point + 1][1]), color=(0, 0, 255), thickness=8)
