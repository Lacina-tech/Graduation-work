import cv2
import numpy
import os

class DataPreprocessing:
    def __init__(self, data):
        self.data = data  # Ukládá obrázek jako instanční proměnnou
            
        # Inicializace Kaskád z OpenCV
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.eyes_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.glasses_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    def detect_faces(self):
        """
        Detekuje obličeje na obrázku pomocí Haar cascade a vrací jen ty, kde byly nalezeny oči nebo brýle.
        """
        # Převeďte obrázek na odstíny šedi
        gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        
        # Detekce obličejů
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        valid_faces = []  # Seznam pro ukládání obličejů s detekcí očí nebo brýlí

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]  # Výřez obličeje v šedém odstínu

            # Pokus o detekci očí v rámci obličeje
            eyes = self.eyes_detector.detectMultiScale(face_region)
            if len(eyes) == 0:
                # Pokud nejsou detekovány oči, pokusí se detekovat brýle
                eyes = self.glasses_detector.detectMultiScale(face_region)

            # Pokud jsou nalezeny oči nebo brýle, přidejte obličej do seznamu validních obličejů
            if len(eyes) > 0:
                valid_faces.append((x, y, w, h, eyes))

        if valid_faces:
            print(f"Počet nalezených obličejů s očima nebo brýlemi: {len(valid_faces)}")
        else:
            print("Žádný obličej s očima nebo brýlemi nebyl nalezen")
        
        return valid_faces

    def draw_faces(self, faces):
        """
        Nakreslí červený obdélník kolem detekovaných obličejů a modré kruhy na místech očí nebo brýlí.
        Přidá nápis "Unknown" pod každým obličejem.
        """
        for (x, y, w, h, eyes) in faces:
            # Vykreslení obdélníku kolem obličeje
            cv2.rectangle(self.data, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Červený obdélník pro obličej

            # Vykreslení kruhů kolem očí nebo brýlí
            for (ex, ey, ew, eh) in eyes:
                # Určení středu každého oka nebo brýlí
                center_x = x + ex + ew // 2
                center_y = y + ey + eh // 2
                radius = max(ew, eh) // 4  # Poloměr kruhu

                # Vykreslení kruhu
                cv2.circle(self.data, (center_x, center_y), radius, (255, 255, 0), 2)  # Světle modrý kruh pro oči nebo brýle

            
        return self.data

    def process_images(self, in_folder=r"c:\Users\uzivatel\Dropbox\MP\Program\test.org", out_folder=r"c:\Users\uzivatel\Dropbox\MP\Program\test.2"):

        input_folder = in_folder
        output_folder = out_folder

        # Ověření existence výstupního adresáře
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        for i in os.listdir(input_folder):
            if i.endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(input_folder, i)
                image = cv2.imread(input_path)

                if image is None:  # Kontrola, zda se obrázek načetl správně
                    print(f"Nelze načíst obrázek: {input_path}")
                    continue

                self.data = image
                faces = self.detect_faces()
                preprocessed_image = self.draw_faces(faces)

                output_path = os.path.join(output_folder, i)
                cv2.imwrite(output_path, preprocessed_image)
                print(f"Obrázek uložen: {output_path}")

if __name__ == "__main__":
    data = None
    ok = DataPreprocessing(data)
    ok.process_images()
    print("Hotovo")