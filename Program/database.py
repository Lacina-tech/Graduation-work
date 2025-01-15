import tensorflow as tf
import sqlite3
import numpy as np
import os

# Implementace modulu
from data_preprocessing import DataPreprocessing
import keras

def load_model():
    keras.config.enable_unsafe_deserialization()
    # Načtení modelu
    model_path = r"Program\epoch_01.keras"
    model = tf.keras.models.load_model(model_path)
    return model

# Funkce pro generování embeddingu
def generate_embedding(model, image):
    # Generování embeddingu pro obrázek
    #image = tf.expand_dims(image, axis=0)
    embedding = model.predict(image)  # Predikce na základě předzpracovaného obrázku
    return embedding

# Funkce pro vytvoření databáze a tabulky
def create_database():
    conn = sqlite3.connect('face_database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            surname TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Funkce pro uložení osoby do databáze
def add_person_to_database(name, surname, images):
    model = load_model()
    conn = sqlite3.connect('face_database.db')
    c = conn.cursor()

    # Předzpracování obrázků
    preprocessed_faces = []
    for i in images:
        preprocessor = DataPreprocessing(i)
        preprocessed_face = preprocessor.preprocess_faces()
        preprocessed_faces.append(preprocessed_face)

    # Generování embeddingů
    embeddings = []
    for face in preprocessed_faces:
        embedding = generate_embedding(model, face)
        embeddings.append(embedding)

    # Průměrování embeddingů
    average_embedding = np.mean(embeddings, axis=0)  # Průměrná hodnota podél dimenzí embeddingů

    # Uložení první generované embedding do databáze
    c.execute('''
        INSERT INTO people (name, surname, embedding) 
        VALUES (?, ?, ?)
    ''', (name, surname, average_embedding.tobytes()))  # Ukládáme embedding jako BLOB

    conn.commit()
    conn.close()

create_database()