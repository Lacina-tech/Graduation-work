import tensorflow

def create_face_recognition_model(input_shape=(128, 128, 3)):
    """
    Vytvoření vlastního modelu pro RO pomocí CNN

    Modul extrahuje rysy obličeje a poté je převede do embeddingu (do matematické reprezentace vektorových vzdáleností), která slouží k rychlému porovnávání a identifikaci obličejů

    Příchozí data:
        parametr: input_shape - určuje tvar vstupních dat (velikost 128x128, má 3 kanály = tnz. RGB formát)
    """
    model = tensorflow.keras.models.Sequential() # Využívá se sekvenční model (vrstvy jsou přidávány v lineárním pořadí - každá výstup vrstvy je vstupem do další vrsty)

    # První konvolunční vrstva

    # Druhá konvolunční vrstva

    # Třetí konvolunční vrstva

    # Vrstva pro vygenerování embeddingu

