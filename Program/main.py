from PyQt5.QtWidgets import QApplication
from gui import MainWindow

def main():
    app = QApplication([])  # Inicializace QApplication bez sys.argv
    window = MainWindow()        # Vytvoření instance MyApp
    window.show()           # Zobrazení okna
    app.exec_()            # Spuštění hlavní smyčky aplikace

if __name__ == '__main__':
    main()