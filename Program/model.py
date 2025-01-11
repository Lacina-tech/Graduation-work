# Modul, který obsahuje kod pro vytvoření a předtrénování modelu, který vytváří embeddingy
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import tensorflow as tf
import numpy as np
import os

# Parametry
IMG_SIZE = (128, 128, 3)
EMBEDDING_SIZE = 128
BATCH_SIZE = 72  # Zvýšeno na větší násobek 3
AUTOTUNE = tf.data.AUTOTUNE

# Funkce pro vytvoření modelu
def build_model(input_shape, embedding_size):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(embedding_size, activation=None)(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return tf.keras.Model(inputs, outputs)

# Vytvoření modelu
model = build_model(IMG_SIZE, EMBEDDING_SIZE)
model.summary()

# Ztrátová funkce - triplet loss
def triplet_loss(y_true, y_pred, margin=0.2):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

# Kompilace modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=triplet_loss)

# Funkce pro předzpracování obrazu
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, IMG_SIZE[:2])

# Funkce pro zajištění násobku 3
def ensure_multiple_of_three(dataset):
    def filter_batch(images, labels):
        # Zjistíme velikost aktuální dávky
        batch_size = tf.shape(images)[0]
        # Pokud není násobkem 3, odstraníme přebytečné prvky
        if batch_size % 3 != 0:
            new_size = batch_size - (batch_size % 3)
            return images[:new_size], labels[:new_size]
        return images, labels

    # Použijeme mapování s více argumenty
    return dataset.map(filter_batch, num_parallel_calls=AUTOTUNE)

# Funkce pro načtení a zpracování datasetu
def load_dataset(data_dir, batch_size, is_training):
    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]  # Extrahování štítku z cesty
        image = preprocess_image(file_path)
        return image, label

    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpeg'), shuffle=is_training)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if is_training:
        dataset = (labeled_ds
                   .shuffle(buffer_size=1000)
                   .batch(batch_size)
                   .prefetch(buffer_size=AUTOTUNE))
    else:
        dataset = (labeled_ds
                   .batch(batch_size)
                   .prefetch(buffer_size=AUTOTUNE))

    # Zajistíme, že dataset obsahuje pouze kompletní trojice
    return ensure_multiple_of_three(dataset)

# Cesty k datům
train_dir = r'Program\dataset\preprocessed dataset (VGGFace2)\train'
val_dir = r'Program\dataset\preprocessed dataset (VGGFace2)\test'

# Načtení datasetů
train_dataset = load_dataset(train_dir, BATCH_SIZE, is_training=True)
val_dataset = load_dataset(val_dir, BATCH_SIZE, is_training=False)

# Callback pro ukládání modelu
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoints/epoch_{epoch:02d}.keras',
    save_best_only=False,  # Ukládá model po každé epoše, ne jen ten nejlepší
    save_weights_only=False,
    verbose=1
)

# Trénování modelu
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=15, 
    verbose=1,
    callbacks=[checkpoint_callback]
)

# Generování embeddings
def generate_embeddings(model, img_path):
    image = preprocess_image(img_path)
    image = tf.expand_dims(image, axis=0)
    embedding = model.predict(image, verbose=1)
    return embedding

# Porovnání embeddingů
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Uložení modelu
model.save('face_recognition_model.h5')