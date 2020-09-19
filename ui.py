from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
import cv2
from main import paintedPoint

class LoginScreen(GridLayout):

    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)

        self.cap = cv2.VideoCapture(0)
        self.capturingCoords = False
        self.paintCoords = []
        self.strokes = []
        self.currentFrame = None

        def mouseMoving(event, x, y, flags, param):
            global capturingCoords, paintCoords
            if event == cv2.EVENT_LBUTTONDOWN:
                self.capturingCoords = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.capturingCoords == True:
                    self.paintCoords.append(paintedPoint('red', x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                self.capturingCoords = False
                self.paintCoords.append(paintedPoint('red', x, y))
                self.strokes.append(self.paintCoords)
                self.paintCoords = []

        cv2.namedWindow(winname="Video")
        cv2.setMouseCallback("Video", mouseMoving)

        i = 0
        while (True):
            _, frame = self.cap.read()
            for point in range(len(self.paintCoords[:-2])):
                cv2.line(frame, pt1=(self.paintCoords[point].x, self.paintCoords[point].y),
                         pt2=(self.paintCoords[point + 1].x, self.paintCoords[point + 1].y),
                         color=(0, 0, 255), thickness=2)
            for stroke in self.strokes:
                for point in range(len(stroke[:-2])):
                    # cv2.circle(frame, (point.x, point.y), 5, (0, 255, 255))
                    cv2.line(frame, pt1=(stroke[point].x, stroke[point].y),
                             pt2=(stroke[point + 1].x, stroke[point + 1].y), color=(0, 0, 255), thickness=2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.imwrite("vid/{}.png".format(i), frame)
            self.vid = Image(source = "vid/{}.png".format(i))
            self.add_widget(self.vid)
            cv2.imshow('Video', frame)
            i += 1

        self.cap.release()
        cv2.destroyAllWindows()



class MyApp(App):

    def build(self):
        return LoginScreen()


if __name__ == '__main__':
    MyApp().run()