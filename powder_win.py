# -*- coding: utf-8 -*-


import os
from functools import partial

import numpy as np
from numpy.linalg import norm
import math
import random
from scipy.stats import linregress
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from util.util import fit_circle, get_photon_energy, get_photon_wavelength, \
    build_grid_image, axis_angle_to_rotation_matrix

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog, QTableWidgetItem


def calc_scattering_angle(peaks_det, det_dist,
                          beam_center, theta_t, phi_t,
                          beam_vector, pixel_size):
    """Calculate scattering angle for given peaks.
    """
    # calculate detector rotation matrix: R_t: n -> n_p
    n_p = np.array([
        np.sin(theta_t) * np.cos(phi_t),
        np.sin(theta_t) * np.sin(phi_t),
        np.cos(theta_t),
    ])
    axis = np.cross(beam_vector, n_p)
    angle = np.arccos(np.dot(beam_vector, n_p) / (norm(beam_vector) * norm(n_p)))
    R_t = axis_angle_to_rotation_matrix(axis, angle)

    # map detector(fs_p, ss_p) pixel coordinates to laboratory coordinates
    fs_p, ss_p = R_t.dot(FS), R_t.dot(SS)
    fs_p = np.tile(fs_p, (peaks_det.shape[0], 1))
    ss_p = np.tile(ss_p, (peaks_det.shape[0], 1))
    peaks = np.zeros((peaks_det.shape[0], 3))
    peaks += np.reshape((peaks_det[:, 0] - beam_center[0])
                        * pixel_size, (-1, 1)) * fs_p
    peaks += np.reshape((peaks_det[:, 1] - beam_center[1])
                        * pixel_size, (-1, 1)) * ss_p
    peaks += np.array([0, 0, det_dist])
    angles = np.rad2deg(np.arccos(
        peaks.dot(beam_vector) / norm(peaks, axis=1) / norm(beam_vector)))
    return angles


def calc_residue(geometry_params, clusters, pixel_size):
    det_dist, center_fs, center_ss, theta_t, phi_t, photon_energy = geometry_params
    wavelength = get_photon_wavelength(photon_energy)
    residue = []
    for cluster in clusters:
        theta = calc_scattering_angle(
            cluster.peaks, det_dist, [center_fs, center_ss], theta_t, phi_t,
            BEAM_VECTOR, pixel_size)
        theta_ref = np.rad2deg(
            2. * np.arcsin(wavelength / (2. * cluster.d_spacing)))
        _residue = np.abs(theta - theta_ref).tolist()
        residue += _residue
    return np.mean(residue)


def calc_residue_with_fixed_photon_energy(geometry_params,
                                          clusters,
                                          photon_energy,
                                          pixel_size):
    det_dist, center_fs, center_ss, theta_t, phi_t = geometry_params
    wavelength = get_photon_wavelength(photon_energy)
    residue = []
    for cluster in clusters:
        theta = calc_scattering_angle(
            cluster.peaks, det_dist, [center_fs, center_ss], theta_t, phi_t,
            BEAM_VECTOR, pixel_size)
        theta_ref = np.rad2deg(
            2. * np.arcsin(wavelength / (2. * cluster.d_spacing)))
        _residue = np.abs(theta - theta_ref).tolist()
        residue += _residue
    return np.mean(residue)


class Cluster(object):
    pass


MAX_PEAKS = 10000
FS = np.array([-1., 0., 0.])  # detector fast scan vector
SS = np.array([0., -1., 0.])  # detector slow scan vector
BEAM_VECTOR = np.array([0., 0., 1.])  # beam travels alone Z axis


class PowderWindow(QWidget):
    def __init__(self, settings):
        super(PowderWindow, self).__init__()
        # load settings
        self.settings = settings
        self.max_peaks = MAX_PEAKS
        self.powder_width = settings.image_width
        self.powder_height = settings.image_height
        self.center = [settings.center_x, settings.center_y]
        self.eps = 3.
        self.min_samples = 100
        self.tol = 10
        self.photon_energy = settings.photon_energy
        self.det_dist = settings.detector_distance * 1E-3
        self.pixel_size = settings.pixel_size * 1E-6

        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/window/powder.ui' % dir_, self)
        self.splitter_2.setSizes([0.4 * self.width(), 0.6 * self.width()])
        self.splitter_3.setSizes([0.4 * self.height(), 0.6 * self.height()])
        self.photonEnergy.setValue(self.photon_energy)
        self.centerX.setValue(self.center[0])
        self.centerY.setValue(self.center[1])
        self.detectorDistance.setValue(self.det_dist * 1000)
        self.epsBox.setValue(self.eps)
        self.minSamples.setValue(self.min_samples)
        self.tolBox.setValue(self.tol)
        self.header_labels = [
            'raw \n peaks num',
            'raw \n radius mean',
            'raw \n radius std/min/max',
            'opt \n peaks num',
            'opt \n radius mean',
            'opt \n radius std/min/max',
            'center/x, y',
            'resolution/Å',
        ]
        self.powder_table.setColumnCount(len(self.header_labels))
        self.powder_table.setHorizontalHeaderLabels(self.header_labels)

        self.full_peaks = np.array([])
        self.peaks = np.array([])
        self.highlight_peaks = np.array([])
        self.highlight_radius = 0.
        self.opt_result = None
        self.refine_result = None

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

        # add plot item to image view
        self.peaks_view.getView().addItem(self.peak_item)
        self.peaks_view.getView().addItem(self.center_item)
        self.peaks_view.getView().addItem(self.highlight_peak_item)
        self.peaks_view.getView().addItem(self.ring_center_item)
        self.peaks_view.getView().addItem(self.highlight_ring_center_item)
        self.update_peaks_view()

        # slots
        self.browseButton.clicked.connect(self.load_peaks)
        self.centerX.valueChanged.connect(
            partial(self.change_center, dim=0)
        )
        self.centerY.valueChanged.connect(
            partial(self.change_center, dim=1)
        )
        self.detectorDistance.valueChanged.connect(self.change_det_dist)
        self.epsBox.valueChanged.connect(self.change_eps)
        self.minSamples.valueChanged.connect(self.change_min_samples)
        self.tolBox.valueChanged.connect(self.change_tol)
        self.optimizeBtn.clicked.connect(self.optimize)
        self.refineBtn.clicked.connect(self.refine)
        self.photonEnergy.valueChanged.connect(self.change_photon_energy)
        self.powder_table.cellClicked.connect(self.highlight_cluster)
        self.powder_table.cellChanged.connect(self.change_resolution)

    def update_settings(self, settings):
        self.powder_width = settings.image_width
        self.powder_height = settings.image_height
        self.center = [settings.center_x, settings.center_y]
        self.center = [settings.center_x, settings.center_y]
        self.photon_energy = settings.photon_energy
        self.det_dist = settings.detector_distance * 1E-3
        self.pixel_size = settings.pixel_size * 1E-6

    @pyqtSlot()
    def load_peaks(self):
        peak_file, _ = QFileDialog.getOpenFileName(
            self, 'Select peak file', '', 'Peak File (*.npz)')
        if len(peak_file) == 0:
            return
        self.peakFile.setText(peak_file)
        self.full_peaks = np.load(peak_file)['powder_peaks']
        if len(self.full_peaks) > self.max_peaks:
            idx = random.sample(
                list(np.arange(len(self.full_peaks))), self.max_peaks)
            self.peaks = self.full_peaks[idx]
        else:
            self.peaks = self.full_peaks
        self.update_peaks_view()

    @pyqtSlot(float)
    def change_center(self, value, dim):
        self.center[dim] = value
        self.update_peaks_view()

    @pyqtSlot(float)
    def change_det_dist(self, det_dist):
        self.det_dist = det_dist * 1E-3

    @pyqtSlot(float)
    def change_photon_energy(self, photon_energy):
        self.photon_energy = photon_energy

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
        self.highlight_peaks = self.opt_result['clusters'][row].peaks
        self.highlight_radius = float(self.powder_table.item(row, 1).text())
        self.highlight_peak_item.setData(pos=self.highlight_peaks + 0.5)
        self.highlight_ring_center_item.setData(
            pos=self.opt_result['clusters'][row].center.reshape(-1, 2) + 0.5)

    @pyqtSlot()
    def optimize(self):
        if self.peaks.size == 0:
            return
        radii = norm(self.peaks - self.center, axis=1)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(radii[:, np.newaxis])
        clusters = []
        wavelength = get_photon_wavelength(self.photon_energy)
        for label in np.unique(db.labels_):
            if label == -1:
                continue
            else:
                cluster = Cluster()
                cluster.label = label
                cluster.peaks = self.peaks[db.labels_ == label]
                radii = norm(cluster.peaks - self.center, axis=1)
                mean_radius = np.mean(radii)
                std_radius = np.std(radii)
                min_radius = np.min(radii)
                max_radius = np.max(radii)
                cluster.raw_nb_peaks = len(cluster.peaks)
                cluster.raw_mean_radius = mean_radius
                cluster.raw_std_radius = std_radius
                cluster.raw_min_radius = min_radius
                cluster.raw_max_radius = max_radius
                res = fit_circle(cluster.peaks, self.center, mean_radius)
                cluster.opt_nb_peaks = res['fitting peaks num']
                cluster.opt_mean_radius = res['radius']
                cluster.opt_std_radius = res['radius_std']
                cluster.opt_min_radius = res['radius_min']
                cluster.opt_max_radius = res['radius_max']
                cluster.center = res['center']
                cluster.theta = np.mean(calc_scattering_angle(
                    cluster.peaks, self.det_dist, self.center,
                    0., 0., BEAM_VECTOR, self.pixel_size))
                cluster.d_spacing = wavelength / np.sin(
                    np.deg2rad(cluster.theta) * 0.5) * 0.5
                clusters.append(cluster)
        clusters.sort(key=lambda c: c.theta)
        # determine phi_t
        cluster_centers = np.array([c.center for c in clusters])
        rough_shift_vec = cluster_centers[0] - cluster_centers[-1]
        rough_shift_vec /= norm(rough_shift_vec)
        slope, intercept, r_value, p_value, std_err = linregress(
            cluster_centers[:, 0], cluster_centers[:, 1]
        )
        fine_shift_vec = np.array([1, slope] / norm([1, slope]))
        if fine_shift_vec.dot(rough_shift_vec) > 0:
            shift_vec = fine_shift_vec
        else:
            shift_vec = -fine_shift_vec
        X_p = np.array([
            np.array([1., 0., 0.]).dot(FS) / norm(FS),
            np.array([1., 0., 0.]).dot(SS) / norm(SS)
        ])  # projected vector on fs/ss plane of X axis of lab ref system
        if np.cross(X_p, shift_vec) > 0:
            phi_t = np.arccos(X_p.dot(shift_vec) / norm(X_p) / norm(shift_vec))
        else:
            phi_t = -np.arccos(
                X_p.dot(shift_vec) / norm(X_p) / norm(shift_vec))
        # determine D*sin(\theta_t)
        cluster_thetas = [c.theta for c in clusters]
        cluster_square_tan_theta = np.tan(np.deg2rad(cluster_thetas)) ** 2
        cluster_center_shift = norm(cluster_centers - cluster_centers[0],
                                    axis=1) * self.pixel_size
        slope, intercept, r_value, p_value, std_err = linregress(
            cluster_square_tan_theta, cluster_center_shift
        )
        D_sin_theta_t = slope
        # determine beam center
        center_fs, center_ss = -intercept / self.pixel_size * shift_vec + \
                               cluster_centers[0]
        theta_t = np.arcsin(D_sin_theta_t / self.det_dist)
        # fitting residue
        residue = calc_residue_with_fixed_photon_energy(
            [self.det_dist, center_fs, center_ss, theta_t, phi_t],
            clusters,
            self.photon_energy, self.pixel_size
        )
        self.opt_result = {
            'center_x': center_fs,
            'center_y': center_ss,
            'det_dist': self.det_dist,
            'theta_t': np.rad2deg(theta_t),
            'phi_t': np.rad2deg(phi_t),
            'clusters': clusters,
            'residue': residue,
        }
        # update optimize results
        self.optDetDist.setText('%.2f' % (self.opt_result['det_dist'] * 1000))
        self.optCenterX.setText('%.2f' % self.opt_result['center_x'])
        self.optCenterY.setText('%.2f' % self.opt_result['center_y'])
        self.optThetaT.setText('%.2f' % self.opt_result['theta_t'])
        self.optPhiT.setText('%.2f' % self.opt_result['phi_t'])
        self.optResidue.setText('%.3e' % self.opt_result['residue'])
        # update table
        self.powder_table.clearContents()
        self.powder_table.setRowCount(0)
        for i in range(len(clusters)):
            c = clusters[i]
            row_dict = {
                'raw \n peaks num': '%d' % c.raw_nb_peaks,
                'raw \n radius mean': '%.2f' % c.raw_mean_radius,
                'raw \n radius std/min/max': '%.2f/%.2f/%.2f'
                    % (c.raw_std_radius, c.raw_min_radius, c.raw_max_radius),
                'opt \n peaks num': '%d' % c.opt_nb_peaks,
                'opt \n radius mean': '%.2f' % c.opt_mean_radius,
                'opt \n radius std/min/max': '%.2f/%.2f/%.2f'
                    % (c.opt_std_radius, c.opt_min_radius, c.opt_max_radius),
                'center/x, y': '%.2f, %.2f'
                    % (c.center[0], c.center[1]),
                'resolution/Å': '%.4f' % c.d_spacing
            }
            self.fill_table_row(row_dict, i)
        # draw cluster centers
        self.ring_center_item.setData(pos=cluster_centers + 0.5)

    @pyqtSlot()
    def refine(self):
        if self.opt_result is None:
            return
        init_params = [
            self.opt_result['det_dist'],
            self.opt_result['center_x'],
            self.opt_result['center_y'],
            np.deg2rad(self.opt_result['theta_t']),
            np.deg2rad(self.opt_result['phi_t']),
            self.photon_energy,
        ]
        if self.fixPhotonEnergy.isChecked():
            res = minimize(
                calc_residue_with_fixed_photon_energy,
                init_params[:-1],
                args=(self.opt_result['clusters'],
                      self.photon_energy,
                      self.pixel_size),
                method='Nelder-Mead')
        else:
            res = minimize(
                calc_residue,
                init_params,
                args=(self.opt_result['clusters'], self.pixel_size),
                method='Nelder-Mead')
        # update refinement results
        self.refiDetDist.setText('%.2f' % (res.x[0] * 1000))
        self.refiCenterX.setText('%.2f' % res.x[1])
        self.refiCenterY.setText('%.2f' % res.x[2])
        self.refiThetaT.setText('%.2f' % np.rad2deg(res.x[3]))
        self.refiPhiT.setText('%.2f' % np.rad2deg(res.x[4]))
        self.refiPhotonEnergy.setText('%.2f' % res.x[5])
        self.refiResidue.setText('%.3e' % res.fun)

    @pyqtSlot(int, int)
    def change_resolution(self, row, col):
        if col != self.header_labels.index('resolution/Å'):
            return
        if self.opt_result is None:
            return
        item = self.powder_table.item(row, col)
        resolution = float(item.text())
        self.opt_result['clusters'][row].d_spacing = resolution
        return

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
        self.center_item.setData(pos=np.array(self.center).reshape(1, 2) + 0.5)
        if len(self.peaks) > 0:
            self.peak_item.setData(pos=self.peaks + 0.5)
