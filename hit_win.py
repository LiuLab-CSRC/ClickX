from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

import os

import numpy as np
import pandas as pd


class HitWindow(QWidget):
    def __init__(self, settings, main_win):
        super(HitWindow, self).__init__()
        # setup layout
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/hit_win.ui' % dir_, self)
        # load settings
        self.settings = settings
        self.main_win = main_win

        self.browseButton.clicked.connect(self.choose_and_load_hits)
        self.table.cellDoubleClicked.connect(self.view_hits)

    @pyqtSlot()
    def choose_and_load_hits(self):
        hit_file, _ = QFileDialog.getOpenFileName(
            self, "Open Hit File", self.settings.workdir, "(*.csv)"
        )
        if len(hit_file) == 0:
            return
        self.hitFile.setText(hit_file)
        self.load_hits(hit_file)

    @pyqtSlot(int, int)
    def view_hits(self, row, _):
        path = self.table.item(row, 0).text()
        dataset = self.table.item(row, 1).text()
        frame = int(self.table.item(row, 2).text())
        self.main_win.load_data(path, dataset=dataset, frame=frame)
        self.main_win.update_file_info()
        self.main_win.change_image()
        self.main_win.update_display()

    def load_hits(self, hit_file):
        df = pd.read_csv(hit_file)
        data = []
        for i in range(len(df)):
            data.append(
                (
                    df['filepath'][i],
                    df['dataset'][i],
                    df['frame'][i],
                    df['nb_peak'][i]
                )
            )
        data = np.array(data, dtype=[
            ('filepath', object),
            ('dataset', object),
            ('frame', int),
            ('nb_peak', int),
        ])
        self.table.setData(data)
