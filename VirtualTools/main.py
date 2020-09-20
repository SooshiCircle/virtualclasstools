import cv2
class paintedPoint:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

class drawingTool:
    def __init__(self):
        self.paintCoords = []
        self.capturingCoords = False
        self.currentX = 0
        self.currentY = 0
        self.rectangleSize = 4

        self.cap = cv2.VideoCapture()
        self.ref, self.frame = self.cap.read()
        for point in self.paintCoords:
            cv2.rectangle(self.frame, pt1= (point.x + self.rectangleSize, point.y + self.rectangleSize), pt2= (point.x + self.rectangleSize, point.y + self.rectangleSize), color=(0, 255, 255),
                      thickness =-1)
        if self.frame != None:
            cv2.imshow("HI", self.frame)

    def mouseMoving(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.capturingCoords = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.capturingCoords:
                self.paintCoords.append(paintedPoint('red', x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.capturingCoords = False
            self.paintCoords.append(paintedPoint('red', x, y))
        cv2.setMouseCallback("Title of Popup Window", self.mouseMoving)

if __name__ == "__main__":
    testing = drawingTool()

