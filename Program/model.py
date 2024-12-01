import tensorflow

def create_face_recognition_model(input_shape=(128, 128,3 )):
    """
    Vytvoření vlastního modelu pro RO pomocí CNN

    Příchozí data:
        parametr: input_shape - určuje tvar vstupních dat (velikost 128x128, má 3 kanály = tnz. RGB formát)
    """