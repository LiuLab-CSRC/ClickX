from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot, QPoint, Qt

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
        self.workdir = self.settings.workdir
        self.main_win = main_win

        self.browse_btn.clicked.connect(self.load_hit_file)
        self.table.cellDoubleClicked.connect(self.view_hit)

    @pyqtSlot()
    def load_hit_file(self):
        print('load hit file')
        hit_file, _ = QFileDialog.getOpenFileName(
            self, "Open Hit File", self.workdir, "(*.csv)"
        )
        if len(hit_file) == 0:
            return
        self.hit_file_le.setText(hit_file)
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

    @pyqtSlot(int, int)
    def view_hit(self, row, _):
        filepath = self.table.item(row, 0).text()
        dataset = self.table.item(row, 1).text()
        frame = int(self.table.item(row, 2).text())
        self.main_win.load_frame(filepath, dataset=dataset, frame=frame)
        self.main_win.update_file_info()
        self.main_win.change_image()
        self.main_win.update_display()
