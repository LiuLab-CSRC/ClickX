import sys

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from PyQt5.uic import loadUi


class GUI(QMainWindow):
    def __init__(self, *args):
        super(GUI, self).__init__(*args)
        loadUi('gui.ui', self)


def main():
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()