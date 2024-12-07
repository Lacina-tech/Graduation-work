# Modul, ve kterém algoritmus pro RO identifikuje a klasifikuje obličeje
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import tensorflow
import numpy

# Implementace modulu
from data_preprocessing import DataPreprocessing

class FaceRecognition:
    def __init__ (self, data=None, model_path="second_face_recognition_model.h5"):
        self.model = tensorflow.keras.models.load_model(model_path)
        self.data = data

        # Pokud existují vstupní data:
        if data is not None:
            preprocessor = DataPreprocessing(data)
            self.preprocessed_faces = preprocessor.preprocess_faces() # Předzpracování vstupních dat
        else:
            self.preprocessed_faces = None
            print("Nejsou vstupní data pro předzpracování")

    def recognize(self):
        """
        Rozpozná obličej ve vstupních datech a použije model pro klasifikaci.
        """
        if self.preprocessed_faces is None or self.detected_faces is None:
            print("Nejsou žádná data pro klasifikaci.")
            return None
        
        recognition_results = []
        labels = []

        for idx, face in enumerate(self.preprocessed_faces):
            # Predikce rozpoznání obličeje
            prediction = self.model.predict(numpy.expand_dims(face, axis=0))  # Přidání dimenze pro batch
            print(f"Predikce pro obličej {idx}: {prediction}")

            predicted_class = numpy.argmax(prediction)  # Třída s nejvyšší pravděpodobností
            confidence = numpy.max(prediction)  # Nejvyšší pravděpodobnost

            print(f"Predikovaná třída: {predicted_class}, Confidence: {confidence}")

            # Pokud je confidence nízká, označíme jako neznámý obličej
            if confidence < 0.0:  # Práh dát na 0.5, zatím snížen kvůli 98% nepřesnosti algoritmu RO
                labels.append("Unknown")
            else:
                labels.append(str(predicted_class))  # Představuje ID osoby, později změnit na jméno

            recognition_results.append(prediction)

        # Ohraničení obličeje a napsání jméno pod obličej
        preprocessor = DataPreprocessing(self.data)
        self.data = preprocessor.draw_faces(self.detected_faces, labels)
        
        return self.data
