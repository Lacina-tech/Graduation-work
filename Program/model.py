import tensorflow


def create_face_recognition_model(input_shape=(128, 128, 3)):
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
        # activation="relu" - 
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
    model.add(tensorflow.keras.layers.Dense(128, activation="relu"))

    # Normalizace embeddingu (pro porovnávání pomocí vektorových vzdáleností)
    model.add(tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1)))

    # Kompilace modelu
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), # Optimalizátor Adam
        loss=tensorflow.keras.losses.CosineSimilarity(axis=1),  # Použití cosine similarity jako ztrátová funkce
        metrics=[tensorflow.keras.metrics.CosineSimilarity(axis=1, name="cosine_similarity")]  # Hodnocení podobnosti embeddingů
    )

    # Vrácení modelu
    return model