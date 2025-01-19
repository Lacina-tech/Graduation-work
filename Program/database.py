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

# Definice triplet loss funkce
def triplet_loss(y_true, y_pred, margin=0.2):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

# Funkce pro načtení modelu
model = None # Glabální proměnná pro model

# Funkce pro načtení modelu
def load_model():
    global model
    if model is None: # Kontrola, jesli už není načten
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
    embedding = model.predict(image)
    return embedding[0]

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
    return faiss.IndexFlatIP(embedding_dim)

# Funkce pro uložení osoby do databáze
def add_person_to_database(name, surname, images, metadata=r"Program\face_metadata.db", faiss_index_path=r"Program\faiss_index.bin"):
    global faiss_index
    
    model = load_model()

    model = load_model()
    conn = sqlite3.connect(metadata)
    c = conn.cursor()

    # Předzpracování obrázků
    preprocessed_faces = []
    for img in images:
        preprocessor = DataPreprocessing(img)
        preprocessed_faces.append(preprocessor.preprocess_faces())

    embeddings = [generate_embedding(model, face) for face in preprocessed_faces]
    average_embedding = np.mean(embeddings, axis=0).astype("float32")

    # Průměrování embeddingů
    average_embedding = np.mean(embeddings, axis=0).astype("float32")  # Průměrná hodnota podél dimenzí embeddingů

    # Přidáníe mebedding do FAISS indexu
    faiss_index.add(np.array([average_embedding]))
    c.execute('INSERT INTO people (name, surname) VALUES (?, ?)', (name, surname))
    conn.commit()
    conn.close()

    # Uložení Faiss indexu na disk
    save_faiss_index(faiss_index, faiss_index_path)

# Funkce pro uložení FAISS indexu na disk
def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)

# Inicializace aplikace
create_database()
embedding_dim = 128
faiss_index_path = r"Program\faiss_index.bin"

if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
else:
    faiss_index = create_faiss_index(embedding_dim)