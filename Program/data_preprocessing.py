import cv2
import numpy
import os

class DataPreprocessing:
    def __init__(self, data):
        # Uložení obrázku jako instanční proměnou
        self.data = data

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

    def crop_faces(self, faces):
        """
        Na základě detekovaných souřadnic z dat se vystřihne obličej
        """
        cropped_faces = []
        for (x1, y1, x2, y2) in faces:
            cropped_faces.append(self.data[y1:y2, x1:x2]) # Oříznutí (X:X = zápis ve tvaru matice)
        return cropped_faces

    def preprocess_faces(self, faces):
        """
        Upraví obličej - normalizace a zmenšení velikosti pro vstup do algoritmu pro RO
        """
        preprocessed_faces = []
        for face in faces:
            # Zmenšení dat na velikost 128x128 pixelů
            face_resized = cv2.resize(face, 128, 128)
            # Normalizace obrázku do rozsahu [0,1]
            face_normalized = face_resized / 255.0 # Děleny číslem 255, jelikož v tomto rozmezí se pohybují hodnoty pixelů (např. RGB)
            preprocessed_faces.append(face_normalized)
        return numpy.array(preprocessed_faces) # Použití numpy pro rychlost a paměťovou efektivnost

    def draw_faces(self, faces):
        """
        Nakreslí zelený obdélník kolem detekovaných obličejů
        """
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(self.data, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Zelený obdélník (BRG)
        return self.data
    # Později přidat téže label se jménem obličeje




# Třída pro předzpracování datasetu (slouží pro trénink datasetu)
class DatasetPreparation:
    def __init__ (self, data_directory):

        self.data_directory = data_directory

    def load_dataset(self):
        """
        Načte obrázky z datasetu
        """
# Prvně vytvořit funkce v DataPreprocessing pro předzpracování, aby bylo z čeho tuto třídu tvořit...