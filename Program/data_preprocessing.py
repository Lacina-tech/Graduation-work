import cv2
import numpy

class DataPreprocessing:
    def __init__(self, data):
        self.data = data  # Ukládá obrázek jako instanční proměnnou

        # Cesta ke klasifikátoru na detekci obličeje
        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.config_path = "deploy.prototxt"

    def detect_faces(self):
        """
        Detekuje obličeje na obrázku pomocí Haar cascade a vrací jen ty, kde byly nalezeny oči nebo brýle.
        """
        # Proměnná, která určuje jak moc se zmenší původní data (Pro zmenšení náročnosti)
        smallator = 1 # Zatím není používán
        # Načtení klasifikátoru
        net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        # Snížení rozlišení vstupního obrázku na polovinu pro větší rychlost
        small_data = cv2.resize(self.data, (self.data.shape[1] // smallator, self.data.shape[0] // smallator))
        h, w = small_data.shape[:2]  # Výška a šířka zmenšeného obrázku
        # Upravení dat pro vyuřití CNN vDNN
        blob = cv2.dnn.blobFromImage(small_data, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        # Souřadnince nalezených obličejů
        detections = net.forward()

        # Vložení nalezených obličejů do seznamu
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Práh důvěry
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h]) * smallator # Výpočet souřadnic umístění obličeje
                faces.append(box.astype("int"))
        return faces

    def draw_faces(self, faces):
        """
        Nakreslí zelený obdélník kolem detekovaných obličejů
        """
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(self.data, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Zelený obdélník (BRG)
        return self.data
