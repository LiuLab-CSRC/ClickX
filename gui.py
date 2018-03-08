import sys
import os

import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QObject
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidgetItem
from PyQt5.uic import loadUi

import numpy as np
from util import *


class GUI(QMainWindow):
    def __init__(self, *args):
        super(GUI, self).__init__(*args)

        # setup layout
        loadUi('gui.ui', self)
        self.splitter.setSizes([0.7*self.width(), 0.3*self.width()])
        self.setAcceptDrops(True)

        self.accepted_file_types = ('lst', 'h5')
        self.files = []
        self.frame = 0
        self.hit_finding_on = True
        self.min_peak_num = 0
        self.min_intensity = 0.
        self.min_distance = 10

        self.peak_item = None

        # setup parameter tree
        params_list = [
            {
                'name': 'File Info', 'type': 'group', 'children': [
                    {'name': 'filepath', 'type': 'str', 'readonly': True},
                    {'name': 'image num', 'type': 'str', 'readonly': True},
                    {'name': 'current image', 'type': 'str', 'readonly': True},
                ]
            },
            {
                'name': 'Basic Operation', 'type': 'group', 'children': [
                    {'name': 'frame', 'type': 'int', 'value': self.frame},
                    {'name': 'hit finding on', 'type': 'bool',
                     'value': self.hit_finding_on}
                ]
            },
            {
                'name': 'Hit Finder Parameters', 'type': 'group', 'children': [
                    {'name': 'min peak num', 'type': 'int', 'value': '10'},
                    {'name': 'min intensity', 'type': 'float',
                        'value': self.min_intensity},
                    {'name': 'min distance', 'type': 'int',
                        'value': self.min_distance}
                ]
            },
        ]
        self.params = Parameter.create(
            name='params', type='group', children=params_list)
        self.hit_finder_tree.setParameters(self.params, showTop=False)

        # signal and slot
        self.file_list.itemDoubleClicked.connect(self.load_file)
        self.line_edit.returnPressed.connect(self.add_file)
        self.params.param('Basic Operation', 'frame').sigValueChanged.connect(
            self.change_frame)
        self.params.param(
            'Basic Operation', 'hit finding on').sigValueChanged.connect(
                self.change_hit_finding)
        self.params.param(
            'Hit Finder Parameters', 'min peak num').sigValueChanged.connect(
                self.change_min_peak_num)
        self.params.param(
            'Hit Finder Parameters', 'min intensity').sigValueChanged.connect(
                self.change_min_intensity)
        self.params.param(
            'Hit Finder Parameters', 'min distance').sigValueChanged.connect(
                self.change_min_distance)

    @pyqtSlot(object, object)
    def change_hit_finding(self, _, hit_finding_on):
        self.hit_finding_on = hit_finding_on
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_peak_num(self, _, min_peak_num):
        self.min_peak_num = min_peak_num
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_intensity(self, _, min_intensity):
        self.min_intensity = min_intensity
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_distance(self, _, min_distance):
        self.min_distance = min_distance
        self.update_display()

    @pyqtSlot()
    def add_file(self):
        self.maybe_add_file(self.line_edit.text())

    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > (len(self.files) - 1):
            frame = len(self.files) - 1
        self.frame = frame
        self.params.param('File Info', 'current image').setValue(
            self.files[self.frame])
        self.params.param('Basic Operation', 'frame').setValue(self.frame)
        self.update_display()

    @pyqtSlot('QListWidgetItem*')
    def load_file(self, file_item):
        print('loading %s' % file_item.text())
        filepath = file_item.text()
        with open(filepath, 'r') as f:
            files = f.readlines()
        for f in files:
            self.files.append(f[:-1])  # remove the last '\n'
        # update file info and display
        self.params.param('File Info', 'filepath').setValue(filepath)
        self.params.param('File Info', 'image num').setValue(len(self.files))
        self.params.param('File Info', 'current image').setValue(
            self.files[self.frame])
        self.update_display()

    def update_display(self):
        img = read_image(self.files[self.frame])
        self.image_view.setImage(img)
        if self.hit_finding_on:
            peaks = eval_image(img, self.min_intensity, self.min_distance)
            if self.peak_item is not None:
                self.peak_item.clear()
            self.peak_item = pg.ScatterPlotItem(
                pos=peaks, symbol='x', pen='r')
            self.image_view.getView().addItem(self.peak_item)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            drop_file = url.toLocalFile()
            file_info = QtCore.QFileInfo(drop_file)
            ext = file_info.suffix()
            if ext in self.accepted_file_types:
                event.accept()
                return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            drop_file = url.toLocalFile()
            self.maybe_add_file(drop_file)

    def maybe_add_file(self, drop_file):
        ext = QtCore.QFileInfo(drop_file).suffix()
        if ext in self.accepted_file_types:
            if os.path.exists(drop_file):
                self.file_list.addItem(drop_file)
            else:
                print('File not exist %s' % drop_file)
        else:
            print('Unaccepted file type: %s' % drop_file)


def main():
    app = QApplication(sys.argv)
    win = GUI()
    win.setWindowTitle('Hit Finder')
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
