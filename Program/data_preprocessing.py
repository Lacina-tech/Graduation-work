# Modul, který předzpracovává a předtrénovává data pro model pro RO - detekce, ořez, redukce množství pixelů a normalizace obličejů
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Import knihoven
import cv2
import numpy
import os

class DataPreprocessing:
    def __init__(self, data):
        # Uložení obrázku jako instanční proměnou
        self.data = data

        # Cesta ke klasifikátoru na detekci obličeje
        self.model_path = r"Program\DNN\res10_300x300_ssd_iter_140000.caffemodel"
        self.config_path = r"Program\DNN\deploy.prototxt"

    def detect_faces(self):
        """
        Detekuje obličeje na obrázku pomocí Haar cascade a vrací jen ty, kde byly nalezeny oči nebo brýle.
        """
        # Proměnná, která určuje jak moc se zmenší původní data (Pro zmenšení náročnosti)
        smallator = 1 # Zatím není používán
        # Načtení klasifikátoru
        net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        # Snížení rozlišení vstupního obrázku na polovinu pro větší rychlost
        small_data = cv2.resize(self.data, (self.data.shape[1] // smallator, self.data.shape[0] // smallator))
        h, w = small_data.shape[:2]  # Výška a šířka zmenšeného obrázku
        # Upravení dat pro vyuřití CNN vDNN
        blob = cv2.dnn.blobFromImage(small_data, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        # Souřadnince nalezených obličejů
        detections = net.forward()

        # Vložení nalezených obličejů do seznamu
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Práh důvěry
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h]) * smallator # Výpočet souřadnic umístění obličeje
                faces.append(box.astype("int"))
        return faces

    def crop_faces(self, faces):
        """
        Na základě detekovaných souřadnic z dat se vystřihne obličej ve čtvercovém rámu
        """
        cropped_faces = []

        # Vytvoření z obdelníku, ve kterém je obličej, čtverec, aby se nedeformoval obličej v pozdější fázi
        for (x1, y1, x2, y2) in faces:
            # Určí velikost stran obdelníku, ve kterém je obličej
            width = x2 - x1
            height = y2 - y1

            # Zjistí, která strana je delší
            max_size = max(width, height)

            # Výpočet nových čtvercových souřadnic (se zarovnáním na střed)
            new_x1 = x1 - (max_size - width) // 2
            new_y1 = y1 - (max_size - height) // 2
            new_x2 = new_x1 + max_size
            new_y2 = new_y1 + max_size

            # Kontrola záporných souřadníc
            if new_x1 < 0:
                new_x2 += - new_x1
                new_x1 = 0
                print(f"Posunutí souřadnic {new_x1}, {new_y1}, {new_x2}, {new_y2}")

            if new_y1 < 0:
                new_y2 += - new_y1
                new_y1 = 0
                print(f"Posunutí souřadnic {new_x1}, {new_y1}, {new_x2}, {new_y2}")


            # Kontrola, že výřez není prázdný
            if new_x2 > new_x1 and new_y2 > new_y1:
                cropped_face = self.data[new_y1:new_y2, new_x1:new_x2]
                if cropped_face.size > 0:
                    cropped_faces.append(cropped_face)
                else:
                    print(f"Nalezen obličej s neplatnými souřadnicemi ({new_x1}, {new_y1}, {new_x2}, {new_y2})")

        return cropped_faces

    def resizing_and_normalizing(self, faces):
        """
        Upraví obličej - normalizace a zmenšení velikosti pro vstup do algoritmu pro RO
        """
        preprocessed_faces = []
        for face in faces:
            if face is None or face.size == 0:  # Kontrola, zda je face validní
                print("Obličem nebyl nalezen (resizing_and_normalizing)")
                continue

            try:
                face_resized = cv2.resize(face, (128, 128))
                face_normalized = face_resized / 255.0  # Normalizace do [0, 1]
                preprocessed_faces.append(face_normalized)
            except cv2.error as e:
                print(f"Chyba při zpracování obličeje: {e}")
                continue

        return numpy.array(preprocessed_faces) # Použití numpy pro rychlost a paměťovou efektivnost

    def preprocess_faces(self):
        """
        Funkce na předzpracování dat pro model RO
        """
        detected_faces = self.detect_faces()
        cropped_faces = self.crop_faces(detected_faces)
        preprocessed_faces = self.resizing_and_normalizing(cropped_faces)
        return preprocessed_faces

    def draw_faces(self, faces):
        """
        Nakreslí zelený obdélník kolem detekovaných obličejů
        """
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(self.data, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Zelený obdélník (BRG)
        return self.data
    # Později přidat téže label se jménem obličeje




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
                output_path = os.path.join(person_directory, f"{photo_original_name}_main_face.jpg")

                # TYTO 2 ŘÁDKY JSOU PROZATIMNÍ
                # De-normalizace před uložením (převod zpět na [0, 255])
                face_denormalized = (largest_face * 255).astype('uint8')               
                


                # Vytvoření cesty pro uložení obrázku
                cv2.imwrite(output_path, face_denormalized)  # Uložení předzpracovaného obličeje
                #numpy.save(output_path, face) # Uložení předzpracovaného obličeje

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