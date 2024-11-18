import cv2
import numpy
import os

class DataPreprocessing:
    def __init__(self, data):
        self.data = data  # Ukládá obrázek jako instanční proměnnou

        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.config_path = "deploy.prototxt"

    def detect_faces(self):
        """
        Detekuje obličeje na obrázku pomocí Haar cascade a vrací jen ty, kde byly nalezeny oči nebo brýle.
        """
        net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        h, w = self.data.shape[:2]
        blob = cv2.dnn.blobFromImage(self.data, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        faces = []
        #confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Práh důvěry
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                faces.append(box.astype("int"))
        return faces

    def draw_faces(self, faces):
        """
        Nakreslí zelený obdélník kolem detekovaných obličejů
        """
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(self.data, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Zelený obdélník (BRG)
        return self.data
