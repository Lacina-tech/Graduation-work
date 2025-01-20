# Modul, který vytváří embeddingy, hledá nejpodobnější embedding vstupnímu embeddingu v databázi, přidává embeddingy do databáze
# Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import tensorflow as tf
import sqlite3
import numpy as np
import faiss
import os
import cv2

# Implementace modulu
from data_preprocessing import DataPreprocessing

# Třída pro normalizaci, kompatibilní s uloženým modelem TensorFlow
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.math.l2_normalize(inputs, axis=1)

# Třída se správou modelu, včetně jeho načítání
class ModelHander:
    def __init__(self, model_path=r"Program\\my_face_recognition_model.h5"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """ Při prvním zavolání načte model"""
        if self.model is None: # Kontrola, jesli už není načten
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                "L2Normalization": L2Normalization, # Registrace vlastní vrstvy
                "triplet_loss": self.triplet_loss   # Registrace vlastní loss funkce
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
        normalized_embedding = tf.math.l2_normalize(embedding, axis=1)  # Zajištění normalizace embeddingu
        return normalized_embedding[0].numpy()

class DatabaseHandler:
    def __init__ (self, embedding_dim=128, metadata_path=r"Program\\database\\face_metadata.db", faiss_index_path=r"Program\\database\\faiss_index.bin"):
        self.embedding_dim = embedding_dim
        self.metadata_path = metadata_path
        self.faiss_index_path = faiss_index_path

        self.create_database() # Vytvoření databáze pro metadata, pokud neexistuje
        self.faiss_index = self.load_or_create_faiss_index() # Načtení nebo vytvoření FAISS index

    def create_database(self):
        """ Vytvoření SQLite databáze pro metadata """
        conn = sqlite3.connect(r'Program\\database\\face_metadata.db')
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
            index = faiss.read_index(self.faiss_index_path)
            return index
        print("Vytvořen nový FAISS index")
        return faiss.IndexFlatIP(self.embedding_dim)  # Použití IP vzdálenosti

    def save_faiss_index(self):
        """ Uložení FAISS indexu na disk """
        faiss.write_index(self.faiss_index, self.faiss_index_path)

    def validate_consistency(self):
        """ Kontrola konzistence mezi SQLite databází a FAISS indexem """
        conn = sqlite3.connect(self.metadata_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM people") # hvězdička :)
        num_records = c.fetchone()[0]
        conn.close()

        num_embeddings = self.faiss_index.ntotal
        print(f"Počet záznamů v databázi: {num_records}, počet embeddingů ve FAISS: {num_embeddings}")

        if num_records != num_embeddings:
            raise ValueError("Neshoda mezi počtem záznamů v databázi a FAISS indexu! Zajistěte konzistenci.")

    # Funkce pro uložení osoby do databáze
    def add_person_to_database(self, name, surname, images, model_handler=ModelHander()):
        """ Přidání osoby do SQLite databáze a FAISS indexu """
        conn = sqlite3.connect(self.metadata_path)
        c = conn.cursor()

        # Kontrola, zda osoba již existuje
        c.execute("SELECT id FROM people WHERE name = ? AND surname = ?", (name, surname))
        result = c.fetchone()
        if result:
            print(f"Osoba {name} {surname} již existuje v databázi se záznamem ID: {result[0]}")
            conn.close()
            return  # Pokud existuje, ukončí přidávání

        # Předzpracování obrázků
        preprocessed_faces = []
        for img in images:
            preprocessor = DataPreprocessing(img)
            preprocessed_faces.extend(preprocessor.preprocess_faces())

        # Generování embeddingů
        embeddings = []
        for face in preprocessed_faces:
            if len(face.shape) == 3:  # Kontrola tvaru obličeje
                face = np.expand_dims(face, axis=0)  # Přidání dimenze batch
            embedding = model_handler.generate_embedding(face)
            embeddings.append(embedding)
        average_embedding = np.mean(embeddings, axis=0).astype("float32")
        print(f"Průměrný embedding uložen do FAISS: {average_embedding}")

        # Přidání embeddingu do FAISS indexu
        self.faiss_index.add(np.array([average_embedding]))

        # Uložení metadat do SQLite databáze
        c.execute("INSERT INTO people (name, surname) VALUES (?, ?)", (name, surname))
        conn.commit()
        conn.close()

        # Uložení FAISS indexu na disk
        self.save_faiss_index()

        # Kontrola konzistence
        self.validate_consistency()

class Matcher:
    def __init__(self, metadata_path=r"Program\\database\\face_metadata.db", model_handler=ModelHander(), faiss_index_path=r"Program\\database\\faiss_index.bin", embedding_dim=128):
        self.metadata_path = metadata_path
        self.model_handler = model_handler
        self.faiss_index_path = faiss_index_path
        self.embedding_dim = embedding_dim

        if os.path.exists(self.faiss_index_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def preprocess_and_embed(self, image):
        """Předzpracuje data a vytvoří embeddingy """
        preprocessor = DataPreprocessing(image)
        preprocessed_faces = preprocessor.preprocess_faces()

        embeddings = []
        for face in preprocessed_faces:
            if len(face.shape) == 3:  # Zajištění správného tvaru
                face = np.expand_dims(face, axis=0)
            embedding = self.model_handler.generate_embedding(face)
            embeddings.append(embedding)
    
        print(f"Vygenerované embeddingy: {len(embeddings)}")
        return embeddings

    def find_closest_in_faiss_index(self, query_embeddings, top_k=1):
        """ Porovnává embeddingy s FAISS indexem """
        query_embeddings = np.array(query_embeddings).astype("float32")
        if query_embeddings.shape[1] != self.faiss_index.d:
            raise ValueError(f"Dimenze dotazů ({query_embeddings.shape[1]}) nesouhlasí s dimenzí indexu ({self.faiss_index.d})")
        
        distances, indices = self.faiss_index.search(query_embeddings, top_k)
        print(f"Vzdálenosti a indexy: {distances}, {indices}")
        return distances, indices

    def get_name_by_index(self, index):
        """ Zjistí jméno osoby podle indexu z SQLite databáze"""
        conn = sqlite3.connect(self.metadata_path)
        c = conn.cursor()
        corrected_index = int(index) + 1  # FAISS index začíná od 0, SQLite ID od 1
        c.execute("SELECT name, surname FROM people WHERE id = ?", (corrected_index,))
        result = c.fetchone()
        conn.close()
        print(result)
        if result:
            print(f"Nalezeno jméno pro index {corrected_index}: {result[0]} {result[1]}")
            return f"{result[0]} {result[1]}"
        print(f"Jméno nenalezeno pro index: {corrected_index}")
        return "Unknown"

    def identify_people(self, image):
        """ Identifikuje všechny osoby na obrázku """
        embeddings = self.preprocess_and_embed(image)

        if not embeddings:
            return []
        
        distances, indices = self.find_closest_in_faiss_index(embeddings, top_k=1)

        identified_names = []
        for i, dist in enumerate(distances):
            if dist[0] > 0.95: # Prahová hodnota
                name = self.get_name_by_index(indices[i][0])
                identified_names.append(name)
            else:
                print(f"Neznámý obličej, podobnost pod prahovou hodnotou: {dist[0]}")
                identified_names.append("Unknown")
        return identified_names

    def draw_faces_with_names(self, image):
        """ Vykreslí okolo obličeje zelený obdelník a napíše k němu jméno osoby"""
        preprocessor = DataPreprocessing(image)
        detected_faces = preprocessor.detect_faces()  # Detekce obličeje
        print(f"Detekované obličeje: {detected_faces}")

        identified_names = self.identify_people(image)  # Identifikace osob
        print(f"Identifikovaná jména: {identified_names}")

        for (x1, y1, x2, y2), name in zip(detected_faces, identified_names):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vykreslení zeleného obdelníku
            cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
