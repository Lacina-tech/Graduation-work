import tensorflow as tf
import sqlite3
import numpy as np
import faiss
import os
from data_preprocessing import DataPreprocessing

# Definice vlastní vrstvy
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.math.l2_normalize(inputs, axis=1)

# Definicemtriplet loss funkce
def triplet_loss(y_true, y_pred, margin=0.2):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

# Funkce pro načtení modelu
def load_model():
    model_path = r"Program\face_recognition_model_fixed_02.h5"  # Ujistěte se, že model existuje
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'L2Normalization': L2Normalization,
            'triplet_loss': triplet_loss
        }
    )
    return model

# Funkce pro generování embeddingu
def generate_embedding(model, image):
    # Obrázek by měl být správně předzpracovaný (rozměry, normalizace)
    embedding = model.predict(image)  # Přidáme batch dimenzi
    return embedding[0]  # Vrátíme 1D embedding

# Funkce pro vytvoření databáze SQLite pro metadata
def create_database():
    conn = sqlite3.connect(r'Program\face_metadata.db')
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

# Funkce pro vytvoření FAISS indexu
def create_faiss_index(embedding_dim):
    index = faiss.IndexFlatIP(embedding_dim) # IP = Inner Product (používá se pro kosinovou podobnost u normalizovaných vektorů)
    return index

# Funkce pro uložení osoby do databáze FAISS a SQLite
def add_person_to_database(name, surname, images, metadata=r"Program\face_metadata.db", faiss_index_path=r"Program\faiss_index.bin"):
    global faiss_index
    
    model = load_model()
    conn = sqlite3.connect(metadata)
    c = conn.cursor()

    # Předzpracování obrázků
    preprocessed_faces = []
    for img in images:
        preprocessor = DataPreprocessing(img)
        preprocessed_face = preprocessor.preprocess_faces()
        preprocessed_faces.append(preprocessed_face)

    # Generování embeddingů
    embeddings = []
    for face in preprocessed_faces:
        embedding = generate_embedding(model, face)
        embeddings.append(embedding)

    # Průměrování embeddingů
    average_embedding = np.mean(embeddings, axis=0).astype("float32")  # Průměrná hodnota podél dimenzí embeddingů

    # Přidáníe mebedding do FAISS indexu
    faiss_index.add(np.array([average_embedding]))

    # Uložení metadat do SQLite
    c.execute('''
        INSERT INTO people (name, surname) 
        VALUES (?, ?)
    ''', (name, surname))  # Ukládáme embedding jako BLOB

    conn.commit()
    conn.close()

    # Uložení Faiss indexu na disk
    save_faiss_index(faiss_index, faiss_index_path)

# Funkce pro uložení Faiss indexu na disk
def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)

# Inicializace databáze
create_database()
embedding_dim = 128
faiss_index = create_faiss_index(embedding_dim)