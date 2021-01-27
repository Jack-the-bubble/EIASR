import cv2
from PyQt5 import QtCore, QtGui, uic, QtWidgets
import sys
import queue
import numpy as np
import threading
import face_extractor as fe
import Codebook as cb
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import Network as ntw
import time


#load gui as xml created in qtCreator
form_class = uic.loadUiType("App.ui")[0] 


capture_thread = None
framesQueue = queue.Queue()
isRunning = False; isStopped = False
frameNum = 0


def ReadFrames(camNum, queue, width, height, fps):
    '''
    Capture camera frames and add them to frames queue

    :param queue - stores 5 last frames
    :params width, height - size of frame to be captured
    :param fps - numebr of frames per second captured by camera
    '''
    global isRunning
    capture = cv2.VideoCapture(camNum)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    
    while(isRunning):
        if isStopped:
            time.sleep(0.020)
            continue
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() >= 5:
            queue.get()
        queue.put(frame)


class CameraImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CameraImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        '''
        Set image to be displayed on widget
        :param image
        '''
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
      
        self.ImgWidget = CameraImageWidget(self.ImgWidget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def start_clicked(self):
        global isRunning
        global isStopped

        if isRunning:
            if isStopped:
                isStopped = False
                self.startButton.setText('Stop')
            else:
                isStopped = True
                self.startButton.setText('Start')

        if not isRunning:
            isRunning = True
            capture_thread.start()
            self.startButton.setText('Stop')
        

    def update_frame(self):
        global frameNum
        if not framesQueue.empty():
            frame = framesQueue.get() # obtain frame from queue
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            self.window_width = self.frameSize().width() * 0.8
            self.window_height = self.frameSize().height() * 0.8
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            
            imc = img[:,140:500] # crop bunds of image from camera to make computation easier
            frameNum = frameNum + 1
            # Estimate and upade displayed angles for every 8th frame
            if frameNum % 8 == 0:
                imf = faceExr._preprocess_cam_img(imc)
                if( imf is not None ):
                    hogV = angleEstHog.Calc_descriptors_Cimg(imf).T
                    h_angle, v_angle = model.predict(hogV)
                    self.lblHorizontalVal.setText('{:.1f}'.format(h_angle))
                    self.lblVerticalVal.setText('{:.1f}'.format(v_angle))
                    print('V: {:.1f}, H: {:.1f}'.format(v_angle, h_angle))

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            # set and display this image in image widget
            self.ImgWidget.setImage(image)

    def closeEvent(self, event):
        global isRunning
        isRunning = False


# Create thread to capture camera frames and add them to framesQueue
capture_thread = threading.Thread(target=ReadFrames, args = (0, framesQueue, 640, 360, 24))

faceExr = fe.FaceExtractor("","")
angleEstHog = cb.Codebook()

model = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
model.model = load_model('saved_model_v2.h5')

app = QtWidgets.QApplication(sys.argv)
w = MainWindowClass(None)
w.setWindowTitle('Webcamera with head orientation estimation')
w.show()
app.exec_()
