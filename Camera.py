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


#load xml created in qtCreator
form_class = uic.loadUiType("App.ui")[0] 


capture_thread = None
framesQueue = queue.Queue()
isRunning = False; isStopped = False
frameNum = 0
save = False
 

def ReadFrames(camNum, queue, width, height, fps):
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
            print (queue.qsize())
        queue.put(frame)


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
        
        #self.window_width = self.ImgWidget.frameSize().width()
        #self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = CameraImageWidget(self.ImgWidget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    #def resizeEvent(self, event):
    #    self.resized.emit()
    #    return super(Window, self).resizeEvent(event)

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
            #self.startButton.setEnabled(False)
            self.startButton.setText('Stop')
        

    def update_frame(self):
        global frameNum
        if not framesQueue.empty():
            frame = framesQueue.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            self.window_width = self.frameSize().width() * 0.8
            self.window_height = self.frameSize().height() * 0.8
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            
            #TODO
            imc = img[:,140:500]
            frameNum = frameNum + 1
            if frameNum % 8 == 0:
                print('Frm: {}'.format(frameNum))
                #self.lblHorizontalVal.setText('{}'.format(frameNum / 24))
                imf = faceExr._preprocess_cam_img(imc)

                if( imf is not None ):
                    #v_angle, h_angle = angleEstHog.Estimate_angles_for_Cimg(imf)
                    hogV = angleEstHog.Calc_descriptors_Cimg(imf).T
                    h_angle, v_angle = model.predict(hogV)
                    self.lblHorizontalVal.setText('{:.1f}'.format(h_angle))
                    self.lblVerticalVal.setText('{:.1f}'.format(v_angle))
                    print('V: {:.1f}, H: {:.1f}'.format(v_angle, h_angle))
                    #if ( (frameNum % 120) == 0 ):
                    #    # cv2.imwrite("test{}.jpg".format(frameNum / 120), imf)
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



capture_thread = threading.Thread(target=ReadFrames, args = (0, framesQueue, 640, 360, 24))
faceExr = fe.FaceExtractor("","")

angleEstHog = cb.Codebook()
angleEstHog.Load_codebook_to_mem('Codebook_cell16_block8_v2')

model = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
model.model = load_model('saved_model.h5')

app = QtWidgets.QApplication(sys.argv)
w = MainWindowClass(None)
w.setWindowTitle('Webcamera with head orientation estimation')
w.show()
app.exec_()
