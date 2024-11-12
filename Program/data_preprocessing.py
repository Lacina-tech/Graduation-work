import cv2
import numpy
import os
import mtcnn


class DataPreprocessing:
    def __init__(self, data):
        self.data = data  # Ukládá obrázek jako instanční proměnnou
            
        # Inicializace Kaskád z OpenCV
        self.video_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self):
        """
        Detekuje obličeje na obrázku pomocí Haar cascade.
        """
        # Převeďte obrázek na odstíny šedi
        gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

        # Detekce obličeje
        faces = self.video_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        return faces

    def draw_faces(self, faces):
        """
        Nakreslí obdélníky kolem detekovaných obličejů na originálním barevném obrázku.
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(self.data, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Modrý obdélník

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
                faces = self.detect_faces_in_photo()
                preprocessed_image = self.draw_faces_in_photo(faces)

                output_path = os.path.join(output_folder, i)
                cv2.imwrite(output_path, preprocessed_image)
                print(f"Obrázek uložen: {output_path}")

if __name__ == "__main__":
    data = None
    ok = DataPreprocessing(data)
    ok.process_images()
    print("Hotovo")