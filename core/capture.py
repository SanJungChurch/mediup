
import cv2

class Camera:
    def __init__(self, cam_id=0, width=1280, height=720, fps=30):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.cam_id} open failed")
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open().")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera.")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
