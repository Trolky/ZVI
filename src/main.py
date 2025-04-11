
import sys
from PyQt5.QtWidgets import QApplication
from gui import ThresholdingApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdingApp()
    window.show()
    sys.exit(app.exec_())
