import os
from functools import partial
import yaml

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog, QTableWidget, \
    QTableWidgetItem


class PowderWindow(QWidget):
    def __init__(self, settings):
        super(PowderWindow, self).__init__()
        # load settings
        self.settings = settings
        self.workdir = settings.workdir
        self.max_peak = settings.max_peak
        self.powder_width = settings.width
        self.powder_height = settings.height
        self.center = settings.center
        self.eps = settings.eps
        self.min_samples = settings.min_samples
        self.tol = settings.tol

        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/powder_win.ui' % dir_, self)
        self.splitter_2.setSizes([0.7 * self.width(), 0.3 * self.width()])
        self.spin_box_0.setValue(self.max_peak)
        self.spin_box_1.setValue(self.powder_width)
        self.spin_box_2.setValue(self.powder_height)
        self.spin_box_3.setValue(int(self.center[0]))
        self.spin_box_4.setValue(int(self.center[1]))
        self.spin_box_5.setValue(self.eps)
        self.spin_box_6.setValue(self.min_samples)
        self.spin_box_7.setValue(self.tol)
        self.header_labels = [
            'label', 'num of points', 'mean', 'std', 'min', 'max'
        ]
        self.powder_table.setHorizontalHeaderLabels(self.header_labels)

        self.full_peaks = np.array([])
        self.peaks = np.array([])
        self.highlight_peaks = np.array([])
        self.highlight_radius = 0.
        self.db = None  # clustering result

        # plot items
        self.peak_item = pg.ScatterPlotItem()
        self.center_item = pg.ScatterPlotItem()
        self.highlight_peak_item = pg.ScatterPlotItem()

        # add plot item to image view
        self.peaks_view.getView().addItem(self.peak_item)
        self.peaks_view.getView().addItem(self.center_item)
        self.peaks_view.getView().addItem(self.highlight_peak_item)

        self.update_peaks_view()

        # slots
        self.browse_btn.clicked.connect(self.load_peaks)
        self.spin_box_0.valueChanged.connect(self.change_max_peak)
        self.spin_box_1.valueChanged.connect(self.change_width)
        self.spin_box_2.valueChanged.connect(self.change_height)
        self.spin_box_3.valueChanged.connect(partial(self.change_center, dim=0))
        self.spin_box_4.valueChanged.connect(partial(self.change_center, dim=1))
        self.spin_box_5.valueChanged.connect(self.change_eps)
        self.spin_box_6.valueChanged.connect(self.change_min_samples)
        self.spin_box_7.valueChanged.connect(self.change_tol)
        self.cluster_btn.clicked.connect(self.do_clustering)
        self.powder_table.cellDoubleClicked.connect(self.highlight_cluster)
        self.fit_btn.clicked.connect(self.do_fitting)

    @pyqtSlot()
    def load_peaks(self):
        peak_file, _ = QFileDialog.getOpenFileName(
            self, "Select peak file", self.workdir, "Peak File (*.peaks)")
        if len(peak_file) == 0:
            return
        self.line_edit_1.setText(peak_file)
        self.full_peaks = np.loadtxt(peak_file)
        if len(self.full_peaks) > self.max_peak:
            self.peaks = self.full_peaks[:self.max_peak]
        else:
            self.peaks = self.full_peaks
        self.update_peaks_view()

    @pyqtSlot(int)
    def change_width(self, width):
        self.powder_width = width
        self.update_peaks_view()

    @pyqtSlot(int)
    def change_height(self, height):
        self.powder_height = height
        self.update_peaks_view()

    @pyqtSlot(float)
    def change_center(self, value, dim):
        self.center[dim] = value
        self.update_peaks_view()

    @pyqtSlot(int)
    def change_max_peak(self, max_peak):
        self.max_peak = max_peak
        if len(self.full_peaks) > max_peak:
            self.peaks = self.full_peaks[:self.max_peak]
        else:
            self.peaks = self.full_peaks
        self.update_peaks_view()

    @pyqtSlot(float)
    def change_eps(self, eps):
        self.eps = eps

    @pyqtSlot(int)
    def change_min_samples(self, min_samples):
        self.min_samples = min_samples

    @pyqtSlot(float)
    def change_tol(self, tol):
        self.tol = tol

    @pyqtSlot()
    def do_clustering(self):
        radii = np.sqrt((self.peaks[:, 0] - self.center[0])**2 +
                        (self.peaks[:, 1] - self.center[1])**2)
        db = DBSCAN(
            eps=self.eps, min_samples=self.min_samples
        ).fit(radii.reshape(-1, 1))
        self.db = db

        labels = set(db.labels_)
        powder_table = self.powder_table
        powder_table.clearContents()
        powder_table.setRowCount(0)
        row = 0
        for label in labels:
            idx = np.where(db.labels_ == label)[0]
            nb_peaks = len(idx)
            radii_mean = radii[idx].mean()
            radii_std = radii[idx].std()
            radii_min = radii[idx].min()
            radii_max = radii[idx].max()
            row_dict = {
                'label': '%d' % label,
                'num of points': '%d' % nb_peaks,
                'mean': '%.2f' % radii_mean,
                'std': '%.2f' % radii_std,
                'min': '%.2f' % radii_min,
                'max': '%.2f' % radii_max
            }
            self.fill_table_row(row_dict, row)
            row += 1
        powder_table.repaint()

    @pyqtSlot(int, int)
    def highlight_cluster(self, row, _):
        label = int(self.powder_table.item(row, 0).text())
        idx = np.where(self.db.labels_ == label)[0]
        self.highlight_peaks = self.peaks[idx]
        self.highlight_radius = float(self.powder_table.item(row, 1).text())
        self.highlight_peak_item.setData(
            pos=self.highlight_peaks + 0.5,
            symbol='+',
            size=10,
            pen='y',
            brush=(255, 255, 255, 0)
        )

    @pyqtSlot()
    def do_fitting(self):
        tol = self.tol
        fitting_peaks = self.highlight_peaks
        while True:
            x, y = fitting_peaks[:, 0], fitting_peaks[:, 1]
            target = lambda args: np.mean(
                (np.sqrt((x - args[0]) ** 2 + (y - args[1]) ** 2) - args[2]) ** 2)
            args = (self.center[0], self.center[1], self.highlight_radius)
            ret = minimize(target, args, method='SLSQP')
            center = ret.x[0:2]
            radii = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            radii_mean = radii.mean()
            radii_std = radii.std()
            outlier = np.abs(radii - radii_mean) >= tol * radii_std
            if sum(outlier) == 0:
                break
            else:
                fitting_peaks = fitting_peaks[~outlier]
        message = '%d peaks used for fitting, final fun %.2f, ' \
                  'center %.2f %.2f, radius %.2f\n' % \
                  (x.size, ret.fun, ret.x[0], ret.x[1], ret.x[2])
        message += 'mean %.2f, std %.2f, min %.2f, max %.2f' % \
                   (radii.mean(), radii.std(), radii.min(), radii.max())
        self.fit_label.setText(message)
        self.fit_label.repaint()

    def fill_table_row(self, row_dict, row):
        row_count = self.powder_table.rowCount()
        if row_count == row:
            self.powder_table.insertRow(row_count)
        for col, field in enumerate(self.header_labels):
            if field not in row_dict.keys():
                continue
            item = self.powder_table.item(row, col)
            if item is None:
                item = QTableWidgetItem(row_dict[field])
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.powder_table.setItem(row, col, item)
            else:
                item.setText(str(row_dict[field]))

    def update_peaks_view(self):
        powder = np.zeros((self.powder_width, self.powder_height))
        powder[0, 0] = 1
        self.peaks_view.setImage(powder)
        self.center_item.setData(
            pos=self.center.reshape(1, 2) + 0.5, symbol='+', size=24,
            pen='g', brush=(255, 255, 255, 0)
        )
        if len(self.peaks) > 0:
            self.peak_item.setData(
                pos=self.peaks + 0.5,
                symbol='o',
                size=5,
                pen='r', brush=(255, 255, 255, 0)
            )