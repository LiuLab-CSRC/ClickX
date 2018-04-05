import os
from functools import partial

import numpy as np
from numpy.linalg import norm
import math
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from util.util import fit_circle, get_photon_energy, get_photon_wavelength, \
    build_grid_image

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog, QTableWidgetItem


class PowderWindow(QWidget):
    def __init__(self, settings):
        super(PowderWindow, self).__init__()
        # load settings
        self.settings = settings
        self.workdir = settings.workdir
        self.max_peaks = settings.max_peaks
        self.powder_width = settings.width
        self.powder_height = settings.height
        self.center = settings.center
        self.eps = settings.eps
        self.min_samples = settings.min_samples
        self.tol = settings.tol
        self.photon_energy = settings.photon_energy
        self.detector_distance = settings.detector_distance
        self.pixel_size = settings.pixel_size

        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/powder_win.ui' % dir_, self)
        self.max_peaks_sb.setValue(self.max_peaks)
        self.width_sb.setValue(self.powder_width)
        self.height_sb.setValue(self.powder_height)
        self.center_x_sb.setValue(int(self.center[0]))
        self.center_y_sb.setValue(int(self.center[1]))
        self.wavelength_sb.setValue(
            get_photon_wavelength(self.photon_energy)
        )
        self.energy_sb.setValue(self.photon_energy)
        self.det_dist_sb.setValue(self.detector_distance)
        self.pixel_size_sb.setValue(self.pixel_size)
        self.eps_sb.setValue(self.eps)
        self.min_samples_sb.setValue(self.min_samples)
        self.tol_sb.setValue(self.tol)
        self.header_labels = [
            'label',
            'raw \n peaks num',
            'raw \n radius mean',
            'raw \n radius std/min/max',
            'fitting \n peaks num',
            'fitting \n radius mean',
            'fitting \n radius std/min/max',
            'center/x, y',
            'resolution/Å',
            'opt det dist'
        ]
        self.powder_table.setColumnCount(len(self.header_labels))
        self.powder_table.setHorizontalHeaderLabels(self.header_labels)

        self.full_peaks = np.array([])
        self.peaks = np.array([])
        self.highlight_peaks = np.array([])
        self.highlight_radius = 0.
        self.db = None  # clustering result
        self.ring_centers = {}
        self.ring_radii = {}
        self.real_radii = {}
        self.ring_2theta = {}
        self.fitting_center = None
        self.ring_labels = []

        # plot items
        self.center_item = pg.ScatterPlotItem(
            symbol='+', size=30, pen='g', brush=(255, 255, 255, 0)
        )
        self.peak_item = pg.ScatterPlotItem(
            symbol='o', size=5, pen='r', brush=(255, 255, 255, 0)
        )
        self.highlight_peak_item = pg.ScatterPlotItem(
            symbol='o', size=10, pen='y', brush=(255, 255, 255, 0)
        )
        self.ring_center_item = pg.ScatterPlotItem(
            symbol='+', size=10, pen='g', brush=(255, 255, 255, 0)
        )
        self.highlight_ring_center_item = pg.ScatterPlotItem(
            symbol='+', size=15, pen='y', brush=(255, 255, 255, 0)
        )
        self.fitting_center_item = pg.ScatterPlotItem(
            symbol='star', size=20, pen='g', brush=(255, 255, 255, 0)
        )
        self.center_line_item = pg.PlotDataItem(
            pen=pg.mkPen('g', width=1, style=Qt.DotLine)
        )

        # add plot item to image view
        self.peaks_view.getView().addItem(self.peak_item)
        self.peaks_view.getView().addItem(self.center_item)
        self.peaks_view.getView().addItem(self.highlight_peak_item)
        self.peaks_view.getView().addItem(self.ring_center_item)
        self.peaks_view.getView().addItem(self.highlight_ring_center_item)
        self.peaks_view.getView().addItem(self.center_line_item)
        self.peaks_view.getView().addItem(self.fitting_center_item)
        self.update_peaks_view()

        # slots
        self.browse_btn.clicked.connect(self.load_peaks)
        self.max_peaks_sb.valueChanged.connect(self.change_max_peaks)
        self.width_sb.valueChanged.connect(self.change_width)
        self.height_sb.valueChanged.connect(self.change_height)
        self.center_x_sb.valueChanged.connect(
            partial(self.change_center, dim=0)
        )
        self.center_y_sb.valueChanged.connect(
            partial(self.change_center, dim=1)
        )
        self.wavelength_sb.valueChanged.connect(self.change_wavelength)
        self.energy_sb.valueChanged.connect(self.change_energy)
        self.det_dist_sb.valueChanged.connect(self.change_det_dist)
        self.pixel_size_sb.valueChanged.connect(self.change_pixel_size)
        self.eps_sb.valueChanged.connect(self.change_eps)
        self.min_samples_sb.valueChanged.connect(self.change_min_samples)
        self.tol_sb.valueChanged.connect(self.change_tol)
        self.fit_btn.clicked.connect(self.cluster_and_fit)
        self.powder_table.cellDoubleClicked.connect(self.highlight_cluster)
        self.powder_table.cellChanged.connect(self.change_resolution)

    @pyqtSlot()
    def load_peaks(self):
        peak_file, _ = QFileDialog.getOpenFileName(
            self, "Select peak file", self.workdir, "Peak File (*.peaks)")
        if len(peak_file) == 0:
            return
        self.peak_file_le.setText(peak_file)
        self.full_peaks = np.loadtxt(peak_file)
        if len(self.full_peaks) > self.max_peaks:
            self.peaks = self.full_peaks[:self.max_peaks]
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

    @pyqtSlot(float)
    def change_wavelength(self, wavelength):
        self.photon_energy = get_photon_energy(wavelength)
        self.energy_sb.setValue(self.photon_energy)

    @pyqtSlot(float)
    def change_energy(self, energy):
        wavelength = get_photon_wavelength(energy)
        self.wavelength_sb.setValue(wavelength)

    @pyqtSlot(float)
    def change_det_dist(self, det_dist):
        self.detector_distance = det_dist

    @pyqtSlot(float)
    def change_pixel_size(self, pixel_size):
        self.pixel_size = pixel_size

    @pyqtSlot(int)
    def change_max_peaks(self, max_peaks):
        self.max_peaks = max_peaks
        if len(self.full_peaks) > max_peaks:
            self.peaks = self.full_peaks[:self.max_peaks]
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

    @pyqtSlot(int, int)
    def highlight_cluster(self, row, _):
        label = int(self.powder_table.item(row, 0).text())
        idx = np.where(self.db.labels_ == label)[0]
        self.highlight_peaks = self.peaks[idx]
        self.highlight_radius = float(self.powder_table.item(row, 1).text())
        self.highlight_peak_item.setData(pos=self.highlight_peaks + 0.5)
        if label in self.ring_labels:
            ring_center = np.array(self.ring_centers[label]).reshape(-1, 2)
            self.highlight_ring_center_item.setData(pos=ring_center + 0.5)

    @pyqtSlot()
    def cluster_and_fit(self):
        if self.peaks.size == 0:
            return
        radii = np.sqrt((self.peaks[:, 0] - self.center[0])**2 +
                        (self.peaks[:, 1] - self.center[1])**2)
        db = DBSCAN(
            eps=self.eps, min_samples=self.min_samples
        ).fit(radii.reshape(-1, 1))
        self.db = db

        labels = list(set(db.labels_))
        powder_table = self.powder_table
        powder_table.clearContents()
        powder_table.setRowCount(0)
        powder_table.clearContents()
        self.ring_labels.clear()
        self.ring_centers.clear()
        self.highlight_ring_center_item.clear()
        self.highlight_peak_item.clear()
        row = 0
        for label in labels:
            # raw cluster parameters
            idx = np.where(db.labels_ == label)[0]
            nb_peaks = len(idx)
            radii_mean = radii[idx].mean()
            radii_std = radii[idx].std()
            radii_min = radii[idx].min()
            radii_max = radii[idx].max()
            raw_dict = {
                'label': '%d' % label,
                'raw \n peaks num': '%d' % nb_peaks,
                'raw \n radius mean': '%.2f' % radii_mean,
                'raw \n radius std/min/max': '%.2f/%.2f/%.2f' %
                                             (radii_std, radii_min, radii_max),
            }
            self.fill_table_row(raw_dict, row)
            # fitting rings
            if label >= 0:  # real cluster
                self.ring_labels.append(label)
                x, y = self.peaks[idx, 0], self.peaks[idx, 1]
                circle = fit_circle(
                    x, y, tol=self.tol,
                    init_center=self.center, init_radius=radii_mean
                )
                fitting_dict = {
                    'fitting \n peaks num': '%d' % circle['fitting peaks num'],
                    'fitting \n radius mean': '%.2f' % circle['radius'],
                    'fitting \n radius std/min/max': '%.2f/%.2f/%.2f' %
                                                     (circle['radius_std'],
                                                      circle['radius_min'],
                                                      circle['radius_max']),
                    'center/x, y': '%.2f, %.2f' %
                    (circle['center'][0], circle['center'][1])
                }
                self.ring_centers[label] = circle['center']
                self.ring_radii[label] = circle['radius']
                self.fill_table_row(fitting_dict, row)
            row += 1
        # add ring centers to peak view
        self.ring_center_item.setData(
            pos=np.array(list(self.ring_centers.values())) + 0.5
        )
        # fit center
        scatter_2theta = []
        ring_centers = []
        ring_radii = []
        for label in self.ring_labels:
            ring_radius = self.ring_radii[label]
            ring_radii.append(ring_radius)
            ring_centers.append(self.ring_centers[label])
            two_theta = math.atan(
                ring_radius * self.pixel_size * 1E-3 / self.detector_distance
            )
            scatter_2theta.append(two_theta)
        ring_centers = np.array(ring_centers)
        square_tan_2theta = np.tan(scatter_2theta) ** 2
        _, x0, _, _, _ = linregress(square_tan_2theta, ring_centers[:, 0])
        _, y0, _, _, _ = linregress(square_tan_2theta, ring_centers[:, 1])
        self.fitting_center = np.array([x0, y0]).reshape(-1, 2)
        self.fitting_center_item.setData(pos=self.fitting_center + 0.5)
        # fit center trend line and plot
        slope, intercept, _, _, _ = linregress(
            ring_centers[:, 0], ring_centers[:, 1])
        if abs(slope) > 1.:
            trend_line_y = np.array(
                [self.fitting_center[0, 1] - 100,
                 self.fitting_center[0, 1] + 100]
            )
            trend_line_x = (trend_line_y - intercept) / slope
        else:
            trend_line_x = np.array(
                [self.fitting_center[0, 0] - 100,
                 self.fitting_center[0, 0] + 100]
            )
            trend_line_y = slope * trend_line_x + intercept
        self.center_line_item.setData(x=trend_line_x, y=trend_line_y)
        # calculate resolution for each ring
        row = 0
        for label in self.ring_labels:
            ring_center = self.ring_centers[label]
            ring_radius = self.ring_radii[label]
            fitting_center = self.fitting_center.reshape(-1)
            center_shift = norm(ring_center - fitting_center)
            real_radius = np.sqrt(ring_radius**2 - center_shift**2)
            self.real_radii[label] = real_radius
            two_theta = math.atan(
                real_radius * self.pixel_size * 1E-3 / self.detector_distance
            )
            self.ring_2theta[label] = two_theta
            wavelength = self.wavelength_sb.value()
            resolution = wavelength / (2. * math.sin(two_theta / 2.))
            res_dict = {
                'resolution/Å': '%.4f' % resolution,
                'opt det dist': '%.2f' % self.detector_distance,
            }
            self.fill_table_row(res_dict, row)
            row += 1
        powder_table.repaint()
        # fitting tilting angle
        ring_centers = []
        ring_2theta = []
        for label in self.ring_labels:
            ring_centers.append(self.ring_centers[label])
            ring_2theta.append(self.ring_2theta[label])
        ring_centers = np.array(ring_centers)
        ring_2theta = np.array(ring_2theta)
        center_shift = norm(
            (ring_centers - ring_centers[-1, :]) * self.pixel_size, axis=1
        )
        slope, _, r_value, _, _ = linregress(
            np.tan(ring_2theta) ** 2., center_shift
        )
        print('r_value square: ', r_value ** 2)
        tilting_angle = np.rad2deg(
            math.asin(slope * 1E-3 / self.detector_distance)
        )
        message = 'beam center: %.2f %.2f, tilting %.3f deg' \
                  % (self.fitting_center[0, 0],
                     self.fitting_center[0, 1],
                     tilting_angle)
        self.fitting_results_lb.setText(message)

    @pyqtSlot(int, int)
    def change_resolution(self, row, col):
        if col != self.header_labels.index('resolution/Å'):
            return
        item = self.powder_table.item(row, col)
        if item is None:
            return
        resolution = float(item.text())
        label = int(self.powder_table.item(row, 0).text())
        wavelength = self.wavelength_sb.value()
        theta = math.asin(wavelength/(2. * resolution))
        ring_radius = self.real_radii[label] * self.pixel_size
        det_dist = ring_radius / math.tan(2.0 * theta) * 1E-3
        det_dist_col = self.header_labels.index('opt det dist')
        item = self.powder_table.item(row, det_dist_col)
        if item is not None:
            item.setText('%.2f' % det_dist)
        self.powder_table.repaint()

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
        powder = build_grid_image(self.powder_width, self.powder_height)
        self.peaks_view.setImage(powder)
        self.peaks_view.setLevels(min=-1, max=2)
        self.center_item.setData(pos=self.center.reshape(1, 2) + 0.5)
        if len(self.peaks) > 0:
            self.peak_item.setData(pos=self.peaks + 0.5)
