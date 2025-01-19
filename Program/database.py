import tensorflow as tf
import sqlite3
import numpy as np
import faiss
import os
from data_preprocessing import DataPreprocessing

# Třída pro normalizaci, kompatibilní s uloženým modelem TensorFlow
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.math.l2_normalize(inputs, axis=1)

# Třída se správou modelu, včetně jeho načítání
class ModelHander:
    def __init__(self, model_path=r"Program\my_face_recognition_model.h5"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """ Při prvním zavolání načte model"""
        if self.model is None: # Kontrola, jesli už není načten
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                'L2Normalization': L2Normalization, # Registrace vlastní vrstvy
                'triplet_loss': self.triplet_loss        # Registrace vlastní loss funkce
                }
            )
        return self.model

    @staticmethod
    def triplet_loss(y_true, y_pred, margin=0.2):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1) # Vzdálenost anchor-positive
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1) # Vzdállenost anchor-negative
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    def generate_embedding(self, image):
        """ Generování embeddingu pomocí načteného modelu """
        model = self.load_model()
        embedding = model.predict(image)
        return embedding[0]

class DatabaseHandler:
    def __init__ (self, embedding_dim=128, metadata_path=r"Program\database\face_metadata.db", faiss_index_path=r"Program\database\faiss_index.bin"):
        self.embedding_dim = embedding_dim
        self.metadata_path = metadata_path
        self.faiss_index_path = faiss_index_path

        self.create_database() # Vytvoření databáze pro metadata, pokud neexistuje
        self.faiss_index = self.load_or_create_faiss_index() # Načtení nebo vytvoření FAISS index

    def create_database(self):
        """ Vytvoření SQLite databáze pro metadata """
        conn = sqlite3.connect(r'Program\database\face_metadata.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                surname TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def load_or_create_faiss_index(self):
        """ Načtení FAISS indexu ze souboru nebo vytvoření nového """
        if os.path.exists(self.faiss_index_path):
            return faiss.read_index(self.faiss_index_path)
        return faiss.IndexFlatIP(self.embedding_dim)

    def save_faiss_index(self):
        """ Uložení FAISS indexu na disk"""
        faiss.write_index(self.faiss_index, self.faiss_index_path)




    # Funkce pro uložení osoby do databáze
    def add_person_to_database(self, name, surname, images, model_handler=ModelHander()):
        """ Přidání osoby do SQLite databáze a FAISS indexu """
        conn = sqlite3.connect(self.metadata_path)
        c = conn.cursor()

        # Předzpracování obrázků
        preprocessed_faces = []
        for img in images:
            preprocessor = DataPreprocessing(img)
            preprocessed_faces.append(preprocessor.preprocess_faces())

        # Generování embeddingů
        embeddings = [model_handler.generate_embedding(face) for face in preprocessed_faces]
        average_embedding = np.mean(embeddings, axis=0).astype("float32")


        # Přidání mebeddingu do FAISS indexu
        self.faiss_index.add(np.array([average_embedding]))

        # Uložení metadat do SQLite databáze
        c.execute("INSERT INTO people (name, surname) VALUES (?, ?)", (name, surname))
        conn.commit()
        conn.close()

        # Uložení FAISS indexu na disk
        self.save_faiss_index()