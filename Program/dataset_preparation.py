# Modul, který předzpracovává dataset, který slouží pro trénování a testováním modelu Ro
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import cv2
import numpy
import os

# Implementace modelu
from data_preprocessing import DataPreprocessing

# Třída pro předzpracování datasetu (slouží pro trénink datasetu)
class DatasetPreparation:
    def __init__ (self, input_dataset_directory, output_dataset_directory):
        
        self.input_dataset_directory = input_dataset_directory
        self.output_dataset_directory = output_dataset_directory

    def load_dataset(self):
        """
        Načte obrázky z datasetu
        """
        # Vytvoření seznamu pro ukládání obrázků a odpovídajícímu labelu (id) osoby
        photos = []
        # Vytvoření slovníku pro označení každé podsložky (osoby)
        person_labels = {}

        # Přiřadí číselných štítků pro osoby dle složek
        for id, person in enumerate(os.listdir(self.input_dataset_directory)):
            person_labels[person] = id # Přiřazení indexu ke každé osobě
            person_directory = os.path.join(self.input_dataset_directory, person)

            # Načtení obrázku ze složky osoby
            for photo_name in os.listdir(person_directory):
                photo_path = os.path.join(person_directory, photo_name)
                # Načtení obrázku pomocí OpenCV
                photo = cv2.imread(photo_path)

                # Přidání obrázku a jeho štítku do seznamu
                photos.append((photo, id, photo_name))

        # Vrácení seznamů
        return photos, person_labels

    def preprocess_and_save_dataset(self):
        """
        Předzpracuje a uloží data
        """
        # Načtení datasetu
        photos, person_labels = self.load_dataset()

        # Pro každý obrázek a jeho label
        for photo, label, photo_name in photos:
            preprocessor = DataPreprocessing(photo)

            # Předzpracování dat
            preprocessed_faces = preprocessor.preprocess_faces()

            # Uložení pouze největšího (hlavního) přezpracovaného obličeje
            if len(preprocessed_faces) > 0:
                # Určujeme největší obličej
                largest_face = max(preprocessed_faces, key=lambda face: face.shape[0] * face.shape[1])

                # Vytvoření výstupní složky pro každý štítek osoby, pokud neexistuje
                person_directory = os.path.join(self.output_dataset_directory, str(label))
                os.makedirs(person_directory, exist_ok=True)

                photo_original_name = os.path.splitext(photo_name)[0]
                output_path = os.path.join(person_directory, f"{photo_original_name}_main_face.npy")

                # TYTO 2 ŘÁDKY JSOU PROZATIMNÍ
                # De-normalizace před uložením (převod zpět na [0, 255])
                #face_denormalized = (largest_face * 255).astype('uint8')               
                


                # Vytvoření cesty pro uložení obrázku
                #cv2.imwrite(output_path, face_denormalized)  # Uložení předzpracovaného obličeje
                numpy.save(output_path, largest_face) # Uložení předzpracovaného obličeje

                # Oznámení o úspěšném uložení
                print(f"Fotka uložena: {output_path}")


        # Informace o dokončení dějě
        print(f"Předzpracovaný dataset byl uložen do složky: {self.output_dataset_directory}")


# DOČASNÝ KÓD PRO EXPERIMENTÁLNÍ PŘEDZPRACOVÁNÍ DATASETU (Bude odstraněno - pravděpodobně přesunuto do vlastního souboru.py)

if __name__ == "__main__":
    # Cesta k původnímu datasetu a složce pro výstup
    input_dataset_directory = r"Program\dataset\original dataset (LFW)\lfw-deepfunneled"
    output_dataset_directory = r"Program\dataset\preprocessed dataset (LFW)"

    dataset_preparation = DatasetPreparation(input_dataset_directory, output_dataset_directory)

    # Předzpracování a uložení datasetu
    dataset_preparation.preprocess_and_save_dataset()
    