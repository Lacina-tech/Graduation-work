# Modul, který obsahuje kód pro vytvoření a předtrénování modelu, který vytváří embeddingy
#   Model používá ztrátovou funkci triplet loss a výstupem jsou normalizované embeddingy
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import tensorflow as tf
import numpy as np
import os
from collections import defaultdict

# Parametry
IMG_SIZE = (128, 128, 3)    # Velikost vstupního obrázku (šířka, výška, počet kanálů)
EMBEDDING_SIZE = 128        # Velikost embeddingu (počet čísel ve výstupu modelu)
BATCH_SIZE = 144            # Počet obrázků v jedné dávce
AUTOTUNE = tf.data.AUTOTUNE # Automatická optimalizace pro načítání dat

# Funkce pro vytvoření modelu
def build_model(input_shape, embedding_size):
    """
    Vytváří CNN model pro generování embeddingů
    Vstup:
        input_shape - vstupní rozměry
        embedding_size - velikost vektorového embeddingu
    Výstup:
        Model vrací normalizované embeddingy
    """
    inputs = tf.keras.Input(shape=input_shape)

    # První konvoluční vrstva
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Druhá konvoluční vrstva
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Třetí konvoluční vrstva
    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Převedení na vektor
    x = tf.keras.layers.Flatten()(x)

    # Převedení na embedding
    x = tf.keras.layers.Dense(embedding_size, activation=None)(x)

    # Normalizace embeddingu na jednotkovou délku
    outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), output_shape=(embedding_size,))(x)
    return tf.keras.Model(inputs, outputs)

# Vytvoření a shrnutí modelu
model = build_model(IMG_SIZE, EMBEDDING_SIZE)
model.summary()

# Ztrátová funkce - triplet loss
def triplet_loss(y_true, y_pred, margin=0.2):
    """
    Funkce triplet loss slouží pro učení vytváření embeddingů. Zajišťuje, že pozitivní příklady jsou blíže k anchor než negativní příklady
    Vstup:
        y_true - není potřeba
        y_pred - embeddingy obsahující anchor, positive a negative
        margin - minimální vzdálenost mezi pozitivním a negativním příkladem
    Výstup:
        Hodnota ztrátové funkce
    """
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1) # Vzdálenost anchor-pozitivní
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1) # Vzdálenost anchor-negativní
    basic_loss = pos_dist - neg_dist + margin # Triplet loss
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0)) # Maximální hodnota mezi 0 a ztrátou

# Funkce pro předzpracování obrazu
def preprocess_image(image_path):
    """ Načtení a "připravení" dat """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, IMG_SIZE[:2])

# Funkce pro výběr tripletů
def triplet_generator(file_paths, labels, batch_size):
    """ Generování tripletů pro trénování modelu """
    label_to_files = defaultdict(list)
    for file_path, label in zip(file_paths, labels):
        label_to_files[label].append(file_path) # Třídění cest pomocí labelů
    
    while True:
        batch_images, batch_labels = [], []
        all_labels = list(label_to_files.keys())
        np.random.shuffle(all_labels) # Promíchání štítků
        
        for label in all_labels:
            pos_files = label_to_files[label]
            if len(pos_files) < 2:
                continue
            
            # Náhodný výběr anchor a pozitivního příkladu
            anchor, positive = np.random.choice(pos_files, size=2, replace=False)
            # Výběr negativního příkladu
            neg_label = np.random.choice([l for l in all_labels if l != label])
            negative = np.random.choice(label_to_files[neg_label])
            
            # Načtení a zpracování obrázků
            for img_path in [anchor, positive, negative]:
                img = preprocess_image(img_path).numpy()
                batch_images.append(img)
                batch_labels.append(label)  # přidání labelu
            
            # Vytvoření dávky
            if len(batch_images) >= batch_size:
                yield (np.array(batch_images[:batch_size], dtype=np.float32), 
                       np.array(batch_labels[:batch_size], dtype=np.int32))
                batch_images, batch_labels = [], []

# Funkce pro načtení cest a štítků
def load_file_paths(data_dir):
    """ Načte cesty k obrázků, a jejich odpovídající labely """
    file_paths, labels = [], []
    class_names = sorted(os.listdir(data_dir))  # Zajištění konzistentního pořadí tříd
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_dir in class_names:
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                file_paths.append(os.path.join(class_path, img_file))
                labels.append(label_map[class_dir])  # Uložení číselného štítku
    return file_paths, labels

# Cesty k trénovacím a validačním datům
train_dir = r"Program\dataset\preprocessed dataset (VGGFace2)\train"
val_dir = r"Program\dataset\preprocessed dataset (VGGFace2)\test"

# Načtení cest a štítků
train_file_paths, train_labels = load_file_paths(train_dir)
val_file_paths, val_labels = load_file_paths(val_dir)

# Generátory dat pro trénování a validaci
train_gen = triplet_generator(train_file_paths, train_labels, BATCH_SIZE)
val_gen = triplet_generator(val_file_paths, val_labels, BATCH_SIZE)

# Počet kroků na epochu
train_steps = len(train_file_paths) // BATCH_SIZE
val_steps = len(val_file_paths) // BATCH_SIZE

# Kompilace modelu s triplet loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=triplet_loss)

# Callback pro ukládání modelu během trénování
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model_checkpoints/epoch_{epoch:02d}.keras",
    save_best_only=False,  # Ukládá model po každé epoše
    save_weights_only=False,
    verbose=1
)

# Trénování modelu
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=1, # Nyní zvolena pouze jedna epocha pro rychlé natrénování modelu
    verbose=1,
    callbacks=[checkpoint_callback]
)

# Uložení natrénovaného modelu
model.save("my_face_recognition_model.h5")
