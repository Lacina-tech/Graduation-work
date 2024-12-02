# Modul, ve kterém algoritmus pro RO identifikuje a klasifikuje obličeje
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import cv2
import tensorflow

class FaceRecognition:
    def __init__(self, model_path):
        self.model = None # Bude načítat model pro RO

    # Funkce na rozpoznávání obličeje
    def recognize_face(self, preprocessed_faces):
        pass