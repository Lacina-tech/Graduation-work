# Modul, který předzpracovává dataset, který slouží pro trénování a testováním modelu RO
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import cv2
import numpy
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# Implementace modelu
from data_preprocessing import DataPreprocessing

# Třída pro předzpracování datasetu (slouží pro trénink datasetu)
class DatasetPreparation:
    def __init__ (self, input_dataset_directory, output_train_dataset_directory, output_test_dataset_directory):
        """
        Inicializace tříd
        """
        self.input_dataset_directory = input_dataset_directory
        self.output_train_dataset_directory = output_train_dataset_directory
        self.output_test_dataset_directory = output_test_dataset_directory

    def load_dataset(self):
        """
        Načte obrázky z datasetu a přiřadí mu unikátní štítek osoby

        funkce vrací 2 objekty
        - photos: seznam, každý prvek obsahuje trojici podprvků (foto, label, jméno_souboru)
        - personal_label: slovník, kde klíč je název složky a hodnota je unikátní štítek osoby
        """
        # Vytvoření seznamu pro ukládání obrázků a odpovídajícímu labelu (id) osoby
        photos = []
        # Vytvoření slovníku pro označení každé složky unikátním štítkem (osoby)
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
        Předzpracovává a ukládá data a rozděluje je na trénovací a testovací sadu
        """
        # Načtení datasetu
        photos, person_labels = self.load_dataset()

        # Extrakce všech štítků z photos
        labels = [label for _, label, _ in photos]
        # Zjištění počtu obrázků v jednotlivé třídě
        label_counts = Counter(labels)

        # Třídy, které obsahují pouze jeden obrázek jsou zařazeny do trénovací sady (train_test_split funguje pouze pro třídy s 2 a více daty)
        train_photos = [photo for photo in photos if label_counts[photo[1]] == 1]
        # Třídy s více obrázky
        multiple_sample_photos = [photo for photo in photos if label_counts[photo[1]] > 1]

        # Rozdělení tříd s více obrázky na trénovací a testovací část v poměru 1:5
        if multiple_sample_photos:
            train_split, test_split = train_test_split(
                multiple_sample_photos, # Data pr orozdělení
                test_size=0.2, # Podíl pro testovací sadu
                stratify=[label for _, label, _ in multiple_sample_photos], # Stratiikace dle tříd
                random_state=42
            )
            # Doplnění trénovací sady o train_split
            train_photos.extend(train_split) 
        else:
            # Testovací sada
            test_split = []

        # Předzpracování a uložení trénovací sady pomocí funkce "process_and_save"
        self.preprocess_faces_and_save(train_photos, self.output_train_dataset_directory, "trénovací")

        # Předzpracování a uložení testovací sady pomocí funkce "process_and_save"
        self.preprocess_faces_and_save(test_split, self.output_test_dataset_directory, "testovací")

    def preprocess_faces_and_save(self, photos, output_directory, name):
        """
        Zpracování a uložení dat do jednotlivých složek dle unikátního štítku
        """
        # Pro každý obrázek a jeho label
        for photo, label, photo_name in photos:
            preprocessor = DataPreprocessing(photo)

            # Předzpracování dat
            preprocessed_faces = preprocessor.preprocess_faces()

            # Uložení pouze největšího (hlavního) přezpracovaného obličeje (ignoruje detekované obličeje na pozadí)
            if len(preprocessed_faces) > 0:
                # Určujeme největší obličej
                largest_face = max(preprocessed_faces, key=lambda face: face.shape[0] * face.shape[1])

                # Vytvoření výstupní složky pro každý štítek osoby, pokud neexistuje
                person_directory = os.path.join(output_directory, str(label))
                os.makedirs(person_directory, exist_ok=True)

                # Pojmenování dat a nastavení formátu data.npy (pro normalizovaná data)
                photo_original_name = os.path.splitext(photo_name)[0]
                output_path = os.path.join(person_directory, f"{photo_original_name}_face.npy")

                # Uložení předzpracovaného datasetu
                numpy.save(output_path, largest_face) # Uložení předzpracovaného obličeje

                # Oznámení o úspěšném uložení
                print(f"Fotka uložena do {name} sady: {output_path}")

        # Informace o dokončení dějě
        print(f"{name} dataset byl uložen do složky: {output_directory}")



if __name__ == "__main__":
    # Cesta k původnímu datasetu a složce pro výstup
    input_dataset_directory = r"Program\dataset\original dataset (LFW)\lfw-deepfunneled"
    output_train_dataset_directory = r"Program\dataset\preprocessed dataset (LFW)\train"
    output_test_dataset_directory = r"Program\dataset\preprocessed dataset (LFW)\test"

    dataset_preparation = DatasetPreparation(input_dataset_directory, output_train_dataset_directory, output_test_dataset_directory)

    # Předzpracování a uložení datasetu
    dataset_preparation.preprocess_and_save_dataset()
    