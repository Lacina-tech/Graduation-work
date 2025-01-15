import tensorflow
import numpy

from data_preprocessing import DataPreprocessing

class FaceRecognition:
    def __init__ (self, data=None, model_path=r"epoch_01.keras", database_path=r"face_database.db"):
        # Načtení modelu
        self.model = tensorflow.keras.models.load_model(model_path)
        # Uložení vstupních dat
        self.data = data

        # Načtení databáze
        self.db_path = database_path
        self.database = self.load_database()

        # Pokud existují vstupní data, předzpracují se
        if data is not None:
            preprocessor = DataPreprocessing(data)
            self.preprocessed_faces = preprocessor.preprocess_faces() # Předzpracování vstupních dat
        else:
            self.preprocessed_faces = None
            print("Nejsou vstupní data pro předzpracování")