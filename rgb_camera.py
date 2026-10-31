import cv2 as cv
from picamera2 import Picamera2

class CAMERA_DRIVER:
    USE_PICAMERA = True
    USB_CAMERA_ID = 0

    def __init__(self, resolution=(640,480), raspCam=True, cameraID=0):
        self.USE_PICAMERA = raspCam
        self.USB_CAMERA_ID = cameraID

        if self.USE_PICAMERA:
            self.picam2 = Picamera2()
            picamconfig = self.picam2.create_video_configuration(main={"size": resolution})
            self.picam2.configure(picamconfig)
            self.picam2.start()

        else:
            self.cam_cap = cv.VideoCapture(self.USB_CAMERA_ID, cv.CAP_V4L2)
            self.cam_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
            self.cam_cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cam_cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cam_cap.set(cv.CAP_PROP_FPS, 30)

            # self.cam_cap = cv.VideoCapture(self.USB_CAMERA_ID)
            # self.cam_cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            # self.cam_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
            # self.cam_cap.set(cv.CAP_PROP_FPS, 30)
            
            if not self.cam_cap.isOpened():
                print("Error: Cannot open camera")
            _ = self.cam_cap.read()

    def capture(self):
        if self.USE_PICAMERA:
            rgb = self.picam2.capture_array()
            frame = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            return True, frame
        else:
            return self.cam_cap.read()

    def close(self):
        if self.USE_PICAMERA:
            self.picam2.stop()
        else:
            self.cam_cap.release()