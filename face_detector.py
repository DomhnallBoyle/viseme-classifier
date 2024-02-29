import os

import cv2
import dlib
import numpy as np

FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

FACIAL_LANDMARKS_IDXS = dict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 35)),
    ('jaw', (0, 17))
])
OFFSETS = {
    'mouth': 15,
}


class FaceDetector:

    def __init__(self, part):
        self.face_detector = cv2.dnn.readNetFromTensorflow(
            os.path.join(DIR_PATH, 'opencv_face_detector_uint8.pb'),
            os.path.join(DIR_PATH, 'opencv_face_detector.pbtxt')
        )
        self.shape_predictor = dlib.shape_predictor(
            os.path.join(DIR_PATH, 'shape_predictor_68_face_landmarks.dat')
        )
        self.part = part

    def _shape_to_np(self, shape, dtype='int'):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def extract_roi(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                shape = \
                    self.shape_predictor(frame, dlib.rectangle(startX, startY, endX, endY))

                shape = self._shape_to_np(shape)

                # loop over the face parts individually
                for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                    if name == self.part:
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                        offset = OFFSETS[self.part]

                        x1 = x - offset
                        y1 = y - offset
                        x2 = (x + w) + offset
                        y2 = (y + h) + offset

                        roi = frame[y1:y2, x1:x2]

                        return roi, x1, y1, x2, y2

        return (None, ) * 5
