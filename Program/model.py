# Modul, který obsahuje vytvořený model pro RO a jeho předtrénování
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihovny
import tensorflow
import numpy as np
import os
import cv2

# Vytvoření vlastní struktury algoritmu pro RO pomocí tensorflow
def create_face_recognition_model(input_shape=(128, 128, 3), num_classes=None):
    """
    Vytvoření vlastního modelu pro RO pomocí CNN

    Modul extrahuje rysy obličeje a poté je převede do embeddingu (do matematické reprezentace vektorových vzdáleností), která slouží k rychlému porovnávání a identifikaci obličejů

    Model potřebuje, aby byla vstupní data normalizována a měla velikost 128x128, 3

    Příchozí data:
        parametr: input_shape - určuje tvar vstupních dat (velikost 128x128, má 3 kanály = tnz. RGB formát)
    """
    model = tensorflow.keras.models.Sequential() # Využívá se sekvenční model (vrstvy jsou přidávány v lineárním pořadí - každá výstup vrstvy je vstupem do další vrsty)

    # První konvolunční vrstva
    # Konvoluční vrstva extrahuje rysy (Conv2D)
        # Model určuje 32 filtrů (rysů), každý rys má velikost 3x3 pixelů
        # activation="relu" - aktivační funkce
        # input_shape=input_shape - určuje velikost vstupních dat v pixelech a počet barevných kanalů
    model.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    # Konvolunční vrstva zmenšuje prostorovou velikost obrazových dat a to ta, že hledá maximální hodnotu určeného okolí (pro snížení náročnosti)
        # (2, 2) - v rozsahu 2x2 pixelů se hledá maximální hodnota
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))

    # Druhá konvolunční vrstva
        # Tato vrstva funguje na stejném principu, ale nyní se hledá 64 filtru - tzn. hledají se složitější rysy
    model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))

    # Třetí konvolunční vrstva
        # Tato vrstva funguje na stejném principu, ale nyní se hledá 128 filtru - tzn. hledají se složitější rysy
    model.add(tensorflow.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))

    # Vrstva pro vygenerování embeddingu
    # Tato vrstva převedení výstupu z předchozí vrstvy (mají maticovou strukturu (2D)) do jednorozměrných vektorů (Flatten = zploštění)
    model.add(tensorflow.keras.layers.Flatten())
    # Tato vrstva zpracovává informace z předchozích vrstev a vytváří komplexní reprezentaci obličeje (Dense)
        # 256 - množství neuronů, které vrstva obsahuje
    model.add(tensorflow.keras.layers.Dense(256, activation="relu"))
    # Tato vrstva dotváří finální obraz obličeje pomocí vektorů, využívá 128 neuronů 
  
    # Výstupní vrstva pro klasifikaci
    if num_classes:
        model.add(tensorflow.keras.layers.Dense(num_classes, activation="softmax"))
    else:
        model.add(tensorflow.keras.layers.Dense(128, activation="relu"))  # Embedding pro porovnání obličejů

    # Kompilace modelu s categorical crossentropy pro více tříd
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Vrácení modelu
    return model

# Funkce pro načtení a přípravu dat
def load_data(dataset_directory):
    data = []
    labels = []
    
    for label in os.listdir(dataset_directory):
        label_path = os.path.join(dataset_directory, label)
        for photo_name in os.listdir(label_path):
            photo_path = os.path.join(label_path, photo_name)
            # Načtení obrázku
            image = cv2.imread(photo_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Převod na RGB
            image = cv2.resize(image, (128, 128))  # Změna velikosti
            image = image.astype("float32") / 255.0  # Normalizace (Bude odstraněna - použita jinde)

            data.append(image)
            labels.append(int(label))

    # Vrácí data a příslučné labely
    return np.array(data), np.array(labels)

# Načtení dat
train_data, train_labels = load_data(r"Program\dataset\preprocessed dataset.1 (LFW)\train")
test_data, test_labels = load_data(r"Program\dataset\preprocessed dataset.1 (LFW)\test")

# Převod štítků na one-hot encoding
num_classes = len(os.listdir(r"Program\dataset\preprocessed dataset.1 (LFW)\train"))
train_labels = tensorflow.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tensorflow.keras.utils.to_categorical(test_labels, num_classes)

# Vytvoření modelu
model = create_face_recognition_model(input_shape=(128, 128, 3), num_classes=num_classes)

# Trénování modelu
history = model.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    epochs=25,
    batch_size=32,
    shuffle=True
)

# Uložení modelu
model.save("face_recognition_model.h5")
print("Model byl úspěšně uložen jako 'face_recognition_model.h5'.")
