import cv2
from PyQt5 import QtCore, QtGui, uic, QtWidgets
import sys
import numpy as np
import threading
import time
import queue

#load xml created in qtCreator
form_class = uic.loadUiType("simple.ui")[0] 


capture_thread = None
framesQueue = queue.Queue()
isRunning = False
frameNum = 0
 

def ReadFrames(camNum, queue, width, height, fps):
    global isRunning
    capture = cv2.VideoCapture(camNum)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(isRunning):
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print (queue.qsize())


class CameraImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CameraImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

class MainWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = CameraImageWidget(self.ImgWidget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    #def resizeEvent(self, event):
    #    self.resized.emit()
    #    return super(Window, self).resizeEvent(event)

    def start_clicked(self):
        global isRunning
        isRunning = True
        capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')


    def update_frame(self):
        global frameNum
        if not framesQueue.empty():
            self.startButton.setText('Stop')
            frame = framesQueue.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            self.window_width = self.frameSize().width()
            self.window_height = self.frameSize().height()
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            #TODO
            imc = img[:,280:1000]
            frameNum = frameNum + 1
            if frameNum % 24 == 0:
                self.lblHorizontalVal.setText('{}'.format(frameNum / 24))
            #END TODO

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def closeEvent(self, event):
        global isRunning
        isRunning = False



capture_thread = threading.Thread(target=ReadFrames, args = (0, framesQueue, 1280, 720, 24))

app = QtWidgets.QApplication(sys.argv)
w = MainWindowClass(None)
w.setWindowTitle('Webcamera with head orientation estimation')
w.show()
app.exec_()
