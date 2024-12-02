# Modul, který obsahuje GUI v PyQt5 pro aplikaci RO
# Autor: Lukáš Lacina 4.B <lacinal@jirovcovka.net>

# Implementace knihoven
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2

# Implementace modelu
from data_preprocessing import DataPreprocessing

class AboutPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("0")
        layout.addWidget(label)

class PhotoUploadPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Vytvoření hlavního layoutu
        layout = QtWidgets.QVBoxLayout(self)

        # Vytvoření layoutu, ve kterém se nachází uživatelem vložená fotka
        # Vytvoření rámu, do kterého se budou náhrávat fotky a jeho omezení
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setStyleSheet("background-color: #b0b0b0;")
        # Nastavení minimální velikosti rámce, která zamezuje rapidnímu růstu okna po vložení fotky s příliš vysokou kvalitou
        self.frame.setMinimumSize(150, 100)
        layout.addWidget(self.frame)
        # Vytvoření labelu, který vkládá obrázek do rámu
        self.image_label = QtWidgets.QLabel(self.frame) # Jeho základem je právě rám
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Vytvoření layoutu, ve kterém se nachází rám s vloženou fotkou a jeho napojení na hlavní layout
        frame_layout = QtWidgets.QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0) # Odstranění okrajů, aby se obrázek napojil až na oraje rámu
        frame_layout.addWidget(self.image_label)

        # Vytvoření layoutu, ve kterém se nachází tlačítka
        # Vytvoření tlačítek
        self.button_load = QtWidgets.QPushButton("Nahrát fotku", self)
        self.button_load.clicked.connect(self.load_image)
        self.button_recognize = QtWidgets.QPushButton("Rozpoznat obličej", self)
        self.button_recognize.clicked.connect(self.face_recognize)
        # Vytvoření layoutu a jeho napojení na hlavní layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_load, 1)
        button_layout.addWidget(self.button_recognize, 2)
        layout.addLayout(button_layout)

        # Proměnné
        ### Uložení nahraného obrázku pro QPixmap a OpenCV ###
        self.loaded_image = None # Pro QPixmap
        self.cv_image = None # Pro OpenCV

    def load_image(self):
        """
        Funkce nahraje obrázek, pokud byl vybrán správný formát
        """
        # Otevře výběr souborů
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Vyber obrázek", "", "Image Files (*.png *.jpg *.bmp)")

        # Pokud byl vložen soubor ve správném formátů, uloží se do proměnné self.loaded_image
        if image_path is not None:
            self.cv_image = cv2.imread(image_path)
            self.loaded_image = QtGui.QPixmap(image_path)
            self.show_image()
        else:
            print("souboor se nenahrál")

    def show_image(self):
        """
        Funkce zobrazí obrázek do rámu a přizpůsobí obrázek velikosti rámu a zachová svůj poměr stran
        """
        if self.loaded_image is not None: # Pokud je vložen obrázek
            frame_size = self.frame.size() 
            scaled_image = self.loaded_image.scaled(frame_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_image)

    def face_recognize (self):
        """
        Funkce, která rozpozná obličej (není definována - zatím je zde jen propojení na úpravu fotky)
        """        
        if self.cv_image is not None:
            # Inicializuje DataPreprocessing s nahraným obrázkem
            processor = DataPreprocessing(self.cv_image)
        
            # Předzpracuje obrázek a detekuje obličeje
            faces = processor.detect_faces()
            edited_image = processor.draw_faces(faces)

            # Změna typz barev z BRG na RGB (OpenCV používá BRG = tzn. bez převodu to změní barvy)
            edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)

            # Převod z OpenCV na QPixmap pro jeho zobrazení
            height, width, channel = edited_image.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(edited_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.loaded_image = QtGui.QPixmap.fromImage(qt_image)  # Uložení upraveného obrázku

            # Zobrazení upraveného obrázku
            self.show_image()
        else:
            print("Není nahraný žádný obrázek.")

    def resizeEvent(self, event):
        """
        Upravuje velikost widgetů podle změny velikosti okna
        """
        super().resizeEvent(event)  # Zavolání rodičovské metody

        # Zjištění velikosti stránky
        page_width = self.size().width()

        # Velikost
        # Pokud obrázek existuje, upraví se jeho velikost při změně okna tak, jako se upravuje samotný rám, ve kterém je fotka vložena
        if self.loaded_image is not None:
            self.show_image()

        # Velikost textu (tlačítka)
        button_font_size = int(page_width // 40)
        for button in [self.button_load, self.button_recognize]:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #24477C;
                    color: white;
                    font-size: {button_font_size}px;
                    font-family: Roboto;
                    font-weight: bold;
                    text-align: center;
                    border-radius: 5px;
                    padding: 7px;
                }}
                QPushButton:hover {{
                    background-color: #0066CC;
                }}
            """)


class LiveRecordingPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Vytvoření hlavního layoutu
        layout = QtWidgets.QVBoxLayout(self)

        # Vytvoření layoutu, ve kterém senachází prostor pro zobrazení kamery
        # Vytvoření rámu, ve kterém se bude zobrazovat video
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setStyleSheet("background-color: #b0b0b0;")
        # Nastavení minimální velikosti rámce, která zamezuje rapidnímu růstu okna po zapnutí kamery
        self.frame.setMinimumSize(150, 100)
        layout.addWidget(self.frame)
        # Vytvoření labelu, který vkládá video do rámu
        self.video_label = QtWidgets.QLabel(self.frame) # Jeho základem je právě rám
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        # Vytvoření layoutu, ve kterém se nachází rám s vloženým videem a jeho napojení na hlavní layout
        frame_layout = QtWidgets.QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0) # Odstranění okrajů, aby se video napojil až na oraje rámu
        frame_layout.addWidget(self.video_label)

        # Vytvoření layoutu, ve kterém se nachází tlačítko
        # Vytvoření tlačítka
        self.button_switching_camera = QtWidgets.QPushButton("Zapnou kameru", self)
        self.button_switching_camera.clicked.connect(self.toggle_camera)
        self.button_face_recognize = QtWidgets.QPushButton("Zapnout rozpoznávání obličeje", self)
        self.button_face_recognize.clicked.connect(self.toggle_face_recognition)
        # Vytvoření layoutu a jeho napojení na hlavní layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_switching_camera, 1)
        button_layout.addWidget(self.button_face_recognize, 2)
        layout.addLayout(button_layout)

        # Proměnné
        self.camera = None
        self.face_recognition_active = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.show_video)

    def toggle_camera(self):
        """
        Umožňuje zapínat a vypínat kameru
        """
        # Pokud je kamera vypnutá, zapne se a spustí se časovač na čtení snímků
        if self.camera is None:  # Pokud je kamera vypnutá
            self.camera = cv2.VideoCapture(0)
            self.timer.start(50)
            self.button_switching_camera.setText("Vypnout kameru")
        # Pokud je kamera vypnutá, vypne se a zastaví časovač + rozpoznávání obličeje
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.video_label.clear()  # Vyčistí label
            self.button_switching_camera.setText("Zapnout kameru")
            self.face_recognition_active = False  # Zastaví rozpoznávání obličeje
            self.button_face_recognize.setText("Zapnout rozpoznávání obličeje")

    def toggle_face_recognition(self):
        """
        Umožňuje zapínat a vypínat rozpoznávání obličeje
        """
        # Pokud je kamera vypnutá, rozpoznávání obličeje není umonžněno
        if self.camera is None:
            print("Kamera není zapnuta. Rozpoznávání obličeje nelze zapnout.")
            return # Ukončení dalšího čtení funkce
        
        self.face_recognition_active = not self.face_recognition_active  # Přepíná stav funkce face_recognition_active
        if self.face_recognition_active:
            print("Rozpoznávání obličeje zapnuto.")  # Diagnostika
            self.button_face_recognize.setText("Vypnout rozpoznávání obličeje")
        else:
            print("Rozpoznávání obličeje vypnuto.")  # Diagnostika
            self.button_face_recognize.setText("Zapnout rozpoznávání obličeje")


    def show_video(self):
        """
        Čte snímky z kamery a zobrazuje je.
        """
        # Pokud je kamera zapnutá, zobrazí se video
        if self.camera is not None: 
            ret, frame = self.camera.read()
            if ret:
                # Snížení rozlišení snímku (např. na 640x480)
                frame = cv2.resize(frame, (640, 480))  # Nové rozlišen

                # Zobrazení snímku a jeho úprava do formátu pro QtPy po zapnutí kamery
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Změna barevného formátu
                self.update_video_label(rgb_frame) # Převedené snímku do příznivého formátu

                # Pokud je zapnuté rozpoznávání obličeje, oznáčení čtverce
                if self.face_recognition_active:
                    # Inicializuje DataPreprocessing s aktuálním snímkem
                    processor = DataPreprocessing(frame)

                    # Detekce obličejů
#                    print("Rozpoznávání obličejů je aktivní.")  # Diagnostika
                    faces = processor.detect_faces()  # Detekce obličejů
                    edited_video = processor.draw_faces(faces)  # Nakreslí obdélníky kolem obličejů

                    # Zobrazení upraveného snímku a jeho úprava do formátu pro QtPy po zapnutí rozpoznávání obličeje
                    edited_video = cv2.cvtColor(edited_video, cv2.COLOR_BGR2RGB)
                    self.update_video_label(edited_video)

    def update_video_label(self, preprocessed_image):
        """
        Upraví snímek do formátu Pixmap z OpenCV, aby se mohl zobrazit v GUI
        """
        frame_size = self.frame.size()
        height, width, channel = preprocessed_image.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(preprocessed_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(frame_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """
        Upravuje velikost widgetů podle změny velikosti okna
        """
        super().resizeEvent(event)  # Zavolání rodičovské metody

        # Zjištění velikosti stránky
        page_width = self.size().width()

        # Velikost
        # Pokud obrázek existuje, upraví se jeho velikost při změně okna tak, jako se upravuje samotný rám, ve kterém je fotka vložena
        if self.camera is not None:
            self.show_video()

        # Velikost textu (tlačítka)
        button_font_size = int(page_width // 40)
        for button in [self.button_switching_camera, self.button_face_recognize]:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #24477C;
                    color: white;
                    font-size: {button_font_size}px;
                    font-family: Roboto;
                    font-weight: bold;
                    text-align: center;
                    border-radius: 5px;
                    padding: 7px;
                }}
                QPushButton:hover {{
                    background-color: #0066CC;
                }}
            """)

class AddFacePage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("3")
        layout.addWidget(label)

# Třída boční lišty
class Sidebar(QtWidgets.QWidget):
    # Konstanty pro barvu lišty 
    SIDEBAR_COLOR = QtGui.QColor(36, 71, 124)

    def __init__(self, stacked_widget):
        super().__init__()

        # Layout pro sidebar
        sidebar_layout = QtWidgets.QVBoxLayout(self)

        # Odkaz na widget pro přepínání obsahu
        self.stacked_widget = stacked_widget

        # Vložení mezery nad hlavičku
        sidebar_layout.addSpacing(20)

        # Vytvoření hlavičky lišty
        # Vložení tématického obrázku a textu
        self.pixmap = QtGui.QPixmap(r"Program\gui_pictures\sidebar\logo_face_recognition.png") # Nahrání obrázku do paměti
        self.icon = QtWidgets.QLabel(self) 
        self.text = QtWidgets.QLabel("Rozpoznávání\nobličeje", self)
        # Vytvoření horizontálního layoutu pro obrázek s textem
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(self.icon)
        header_layout.addWidget(self.text)
        header_layout.setAlignment(QtCore.Qt.AlignCenter) # Zarovnání hlavičky
        # Vložení hlavičky do lišty
        sidebar_layout.addLayout(header_layout)

        # Vložení mezery mezi hlavičku a tlačítka
        sidebar_layout.addSpacing(40)

        # Vytvoří tlačítka
        self.sidebar_button_about = self.create_sidebar_button("   O Aplikaci", r"Program\gui_pictures\sidebar\about.png", 0)
        self.sidebar_button_photo_upload = self.create_sidebar_button("   Nahrát Fotku",r"Program\gui_pictures\sidebar\photo_upload.png", 1)
        self.sidebar_button_live_recording = self.create_sidebar_button("   Živé Snímání",r"Program\gui_pictures\sidebar\live_recording.png", 2)
        self.sidebar_button_add_face = self.create_sidebar_button("   Přidat Obličej",r"Program\gui_pictures\sidebar\add_face.png", 3)

        # Umístí tlačítka do lišty
        sidebar_layout.addWidget(self.sidebar_button_about)
        sidebar_layout.addWidget(self.sidebar_button_photo_upload)
        sidebar_layout.addWidget(self.sidebar_button_live_recording)
        sidebar_layout.addWidget(self.sidebar_button_add_face)

        # Vyplní prostor pod tlačítky, diky čemu jsou tlačítka umístěna nahoře vedle sebe a nejsou roztáhlá po celé liště
        sidebar_layout.addStretch()
        
        # Nastavení barvy pomocí QPalette (setStyleSheet nefungoval)
        p = self.palette()
        p.setColor(self.backgroundRole(), self.SIDEBAR_COLOR)  # Nastaví modrou barvu
        self.setPalette(p)
        self.setAutoFillBackground(True)  # Zajistí, že se pozadí vyplní barvou

    def create_sidebar_button(self, text, icon, page_index):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(page_index))
        button.setIcon (QtGui.QIcon(icon))
        return button
    
    def resizeEvent(self, event):
        # Dynamické škálování textu a obrázku podle šířky sidebaru
        sidebar_width = self.size().width()

        # Zvětšování hlavičky společně s lištou
        # Velikosti textu (hlavička)
        font_size = int(sidebar_width // 12)
        self.text.setStyleSheet(f"color: white; font-size: {font_size}px;")
        # Velikosti obrázku (hlavička)
        icon_width = int(sidebar_width // 3)
        scaled_pixmap = self.pixmap.scaled(icon_width, icon_width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.icon.setPixmap(scaled_pixmap)

        # Zvětšování tlačítek společně s lištou
        # Velikost ikon (tlačítka)
        icon_size = QtCore.QSize(int(sidebar_width // 8), int(sidebar_width // 8))
        for button in [self.sidebar_button_about, self.sidebar_button_photo_upload, self.sidebar_button_live_recording, self.sidebar_button_add_face]:
            button.setIconSize(icon_size)
        # Velikost textu (tlačítka)
        button_font_size = int(sidebar_width // 12)
        for button in [self.sidebar_button_about, self.sidebar_button_photo_upload, self.sidebar_button_live_recording, self.sidebar_button_add_face]:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.SIDEBAR_COLOR.name()};
                    color: white;
                    font-size: {button_font_size}px;
                    font-family: Roboto;
                    font-weight: bold;
                    text-align: left;
                    border-radius: 5px;
                    padding: 7px;
                }}
                QPushButton:hover {{
                    background-color: #0066CC;
                }}
            """)

# Třída hlavního okna
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Nastavení velikosti a titulku GUI
        self.setWindowTitle("Rozpoznávání obličeje")
        self.setGeometry(300, 200, 1280, 720)
        
        self.init_gui()

    def init_gui(self):
        # Hlavní widget a layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # Hlavní rozvržení
        layout = QtWidgets.QHBoxLayout(central_widget)

        # Vytvoření stránky pro každé tlačítko na liště pomocí QStackedWidget
        self.stacked_widget = QtWidgets.QStackedWidget()
        # Jednotlivé stránky (napojení na jejich třídu)
        self.stacked_widget.addWidget(AboutPage()) # Strana 0
        self.stacked_widget.addWidget(PhotoUploadPage()) # Strana 1
        self.stacked_widget.addWidget(LiveRecordingPage()) # Strana 2 
        self.stacked_widget.addWidget(AddFacePage()) # Strana 3

        # Vytvoření boční lišty
        sidebar_widget = Sidebar(self.stacked_widget)
        layout.addWidget(sidebar_widget, 1)

        # Vytvoření přepínání obsahu stránky (za pomocí lišty) pomocí štosování widgetů
        layout.addWidget(self.stacked_widget, 4)
