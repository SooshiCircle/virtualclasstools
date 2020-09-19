import cv2
from main import paintedPoint

type = "hand"
cap = cv2.VideoCapture(0)
capturingCoords = False
paintCoords = []
strokes = []

if type == "mouse":
    def mouseMoving(event, x, y, flags, param):
        global capturingCoords, paintCoords
        if event == cv2.EVENT_LBUTTONDOWN:
            capturingCoords = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if capturingCoords == True:
                paintCoords.append(paintedPoint('red', x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            capturingCoords = False
            paintCoords.append(paintedPoint('red', x, y))
            strokes.append(paintCoords)
            paintCoords = []
        print(paintCoords)
    cv2.namedWindow(winname = "Video")
    cv2.setMouseCallback("Video", mouseMoving)

    while(True):
        _, frame = cap.read()
        for point in range(len(paintCoords[:-2])):
            cv2.line(frame, pt1=(paintCoords[point].x, paintCoords[point].y), pt2=(paintCoords[point + 1].x, paintCoords[point + 1].y),
                     color=(0, 0, 255), thickness=2)
        for stroke in strokes:
            for point in range(len(stroke[:-2])):
                #cv2.circle(frame, (point.x, point.y), 5, (0, 255, 255))
                cv2.line(frame, pt1=(stroke[point].x, stroke[point].y), pt2 = (stroke[point +1].x, stroke[point+1].y), color = (0, 0, 255), thickness= 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Video', frame)
elif type == "hand":


cap.release()
cv2.destroyAllWindows()
