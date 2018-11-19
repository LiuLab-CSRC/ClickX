# -*- coding: utf-8 -*-

"""
Usage:
    gui.py startproject <project> -f <facility>
    gui.py
    gui.py -h | --help
    gui.py --version

Options:
    -h --help                   Show this screen.
    --version                   Show version.
    -f --facility=<facility>    Specify facility e.g. LCLS/PAL/local
                                [default: local].
"""

import os
import sys
from docopt import docopt
from datetime import datetime
from functools import partial
from glob import glob
from shutil import copyfile, copytree

import h5py
import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, QPoint, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidgetItem, \
    QDialog, QFileDialog, QMenu, QTableWidgetItem, QWidget, QDialogButtonBox
from PyQt5.uic import loadUi
from pyqtgraph.parametertree import Parameter

from stats_viewer import StatsViewer
from job_win import JobWindow
from powder_win import PowderWindow
from settings import Settings, SettingDialog
from threads import MeanCalculatorThread, GenPowderThread
from util import util
from util import geometry


SOURCE_DIR = os.path.dirname(__file__)


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        # setup layout
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/gui.ui' % dir_, self)
        self.dataset_diag = QDialog()
        loadUi('%s/ui/dialogs/dataset.ui' % dir_, self.dataset_diag)
        self.setting_diag = SettingDialog()
        self.inspector = QDialog()
        self.inspector.setWindowFlags(
            self.inspector.windowFlags() | Qt.WindowStaysOnTopHint)
        loadUi('%s/ui/widgets/inspector.ui' % dir_, self.inspector)
        self.peak_table = QWidget()
        loadUi('%s/ui/widgets/peak_table.ui' % dir_, self.peak_table)
        self.mean_diag = QDialog()
        loadUi('%s/ui/dialogs/mean_sigma.ui' % dir_, self.mean_diag)
        self.proxy_diag = QDialog()
        loadUi('%s/ui/dialogs/proxy_data.ui' % dir_, self.proxy_diag)
        self.powder_diag = QDialog()
        loadUi('%s/ui/dialogs/powder.ui' % dir_, self.powder_diag)
        self.splitter.setSizes([0.2 * self.height(),
                                0.8 * self.height()])
        self.splitter_2.setSizes([0.3 * self.height(),
                                  0.2 * self.height(),
                                  0.5 * self.height()])
        self.splitter_3.setSizes([0.2 * self.width(),
                                  0.5 * self.width(),
                                  0.3 * self.width()])
        self.splitter_4.setSizes([0.5 * self.width(), 0.5 * self.width(), 0])
        self.maskView.hide()
        self.debugView.hide()
        self.show_file_list = True
        self.show_mask_view = False
        self.show_debug_view = False

        # load settings
        self.settings = Settings(self.setting_diag)
        self.settings.save_settings()

        # other windows
        self.job_win = JobWindow(main_win=self, settings=self.settings)
        self.stats_win = StatsViewer(main_win=self, settings=self.settings)
        self.powder_win = PowderWindow(settings=self.settings)

        # fixed attributes
        self.accepted_file_types = ('h5', 'npy', 'cxi', 'npz', 'lcls', 'tif')
        self.hit_finders = ('snr model', 'poisson model')
        self.peak_refine_mode_list = ('gradient', 'mean')
        self.snr_mode_list = ('adaptive', 'simple', 'rings')

        self.all_files = []  # available files in file list
        self.path = None  # path of current file
        self.mask_file = None
        self.mask = None
        self.geom_file = None
        self.geom = None
        self.eraser_mask = None
        self.h5_obj = None  # h5 object
        self.lcls_data = {}  # lcls data structure
        self.dataset = ''  # current dataset
        self.total_frames = 0  # total frames of current dataset
        self.raw_image = None  # current raw image for display
        self.shape = None
        self.new_shape = None  # new shape after geometry applied
        self.mask_image = None
        self.debug_image = None
        self.peak_info = None
        self.strong_peaks = None
        self.thres_map = None
        # viewer parameters
        self.curr_frame = 0  # current frame
        self.show_center = True
        self.apply_geom = False
        self.auto_range = False
        self.auto_level = False
        self.auto_histogram_range = False
        # calib/mask parameters
        self.center = np.array([720, 720])
        self.mask_thres = 0.
        self.ring_radii = np.array([])
        self.erosion1_size = 0
        self.dilation_size = 0
        self.erosion2_size = 0
        self.show_eraser = False
        self.eraser_size = 100
        self.enable_eraser = False
        # hit finding parameters
        self.hit_finding_on = False
        self.mask_on = False
        self.hit_finder = self.hit_finders[0]
        self.max_peaks = 500
        self.min_peaks = 20
        self.adu_per_photon = 20  # dummy number
        self.epsilon = 1E-5
        self.bin_size = 4
        self.gaussian_sigma = 1.0
        self.min_gradient = 10.
        self.min_distance = 10
        self.merge_flat_peaks = False
        self.crop_size = 7
        self.peak_refine_mode = self.peak_refine_mode_list[0]
        self.min_snr = 6.  # min srn for a strong peak
        self.min_pixels = 2  # min pixel num for a strong peak
        self.max_pixels = 10  # max pixel num for a strong peak
        self.snr_mode = self.snr_mode_list[0]
        self.bg_ratio = 0.7
        self.sig_ratio = 0.2
        self.sig_radius = 1
        self.bg_inner_radius = 2
        self.bg_outer_radius = 3
        self.sig_thres = 5.
        self.show_raw_peaks = False
        self.show_valid_peaks = False
        self.show_opt_peaks = False
        self.show_strong_peaks = True
        # plot items
        self.raw_peak_item = pg.ScatterPlotItem(
            symbol='x', size=10, pen='r', brush=(255, 255, 255, 0)
        )
        self.valid_peak_item = pg.ScatterPlotItem(
            symbol='t', size=10, pen='c', brush=(255, 255, 255, 0)
        )
        self.opt_peak_item = pg.ScatterPlotItem(
            symbol='+', size=10, pen='y', brush=(255, 255, 255, 0)
        )
        self.strong_peak_item = pg.ScatterPlotItem(
            symbol='o', size=10, pen='g', brush=(255, 255, 255, 0)
        )
        self.raw_center_item = pg.ScatterPlotItem(
            symbol='+', size=24, pen='g', brush=(255, 255, 255, 0))
        self.mask_center_item = pg.ScatterPlotItem(
            symbol='+', size=24, pen='g', brush=(255, 255, 255, 0))
        self.raw_ring_item = pg.ScatterPlotItem(
            symbol='o',
            pen=pg.mkPen(width=3, color='y', style=QtCore.Qt.DotLine),
            brush=(255, 255, 255, 0), pxMode=False)
        self.mask_ring_item = pg.ScatterPlotItem(
            symbol='o',
            pen=pg.mkPen(width=3, color='y', style=QtCore.Qt.DotLine),
            brush=(255, 255, 255, 0), pxMode=False)
        self.eraser_item = pg.CircleROI(pos=self.center, size=self.eraser_size)
        if not self.show_eraser:
            self.eraser_item.hide()
        self.rawView.view.addItem(self.raw_peak_item)
        self.rawView.view.addItem(self.valid_peak_item)
        self.rawView.view.addItem(self.opt_peak_item)
        self.rawView.view.addItem(self.strong_peak_item)
        self.rawView.view.addItem(self.raw_center_item)
        self.rawView.view.addItem(self.raw_ring_item)
        self.maskView.view.addItem(self.mask_center_item)
        self.maskView.view.addItem(self.mask_ring_item)
        self.rawView.view.addItem(self.eraser_item)
        # threads and timers
        self.calc_mean_thread = None
        self.gen_powder_thread = None
        # status tree
        status_params = [
            {
                'name': 'filepath', 'type': 'str', 'readonly': True,
                'value': self.path,
            },
            {
                'name': 'dataset', 'type': 'str', 'readonly': True,
                'value': self.dataset,
            },
            {
                'name': 'mask file', 'type': 'str', 'readonly': True,
                'value': self.mask_file,
            },
            {
                'name': 'geom file', 'type': 'str', 'readonly': True,
                'value': self.geom_file,
            },
            {
                'name': 'shape', 'type': 'str', 'readonly': True,
                'value': self.shape,
            },
            {
                'name': 'total frame', 'type': 'str', 'readonly': True,
                'value': self.total_frames,
            },
        ]
        self.status_params = Parameter.create(
            name='status parameters',
            type='group',
            children=status_params,
        )
        self.statusTree.setParameters(self.status_params, showTop=False)
        # viewer tree
        viewer_params = [
            {
                'name': 'current frame', 'type': 'int',
                'value': self.curr_frame,
            },
            {
                'name': 'show center', 'type': 'bool',
                'value': self.show_center,
            },
            {
                'name': 'apply geom', 'type': 'bool',
                'value': self.apply_geom,
            },
            {
                'name': 'display', 'type': 'group', 'children': [
                    {
                        'name': 'auto range', 'type': 'bool',
                        'value': self.auto_range,
                    },
                    {
                        'name': 'auto level', 'type': 'bool',
                        'value': self.auto_level,
                    },
                    {
                        'name': 'auto hist range', 'type': 'bool',
                        'value': self.auto_histogram_range,
                    }
                ],
                'expanded': False,
            }
        ]
        self.viewer_params = Parameter.create(
            name='viewer parameters',
            type='group',
            children=viewer_params,
        )
        self.viewerTree.setParameters(self.viewer_params, showTop=False)
        # calib/mask parameter tree
        calib_mask_params = [
            {
                'name': 'threshold', 'type': 'float',
                'value': self.mask_thres,
            },
            {
                'name': 'center x', 'type': 'float', 'value': self.center[0]
            },
            {
                'name': 'center y', 'type': 'float', 'value': self.center[1]
            },
            {
                'name': 'radii of rings', 'type': 'str', 'value': ''
            },
            {
                'name': 'morphology', 'type': 'group', 'children': [
                    {
                        'name': 'erosion1 size', 'type': 'int',
                        'value': self.erosion1_size
                    },
                    {
                        'name': 'dilation size', 'type': 'int',
                        'value': self.dilation_size
                    },
                    {
                        'name': 'erosion2 size', 'type': 'int',
                        'value': self.erosion2_size
                    },
                ],
                'expanded': False,
            },
            {
                'name': 'eraser', 'type': 'group', 'children': [
                    {
                        'name': 'show', 'type': 'bool',
                        'value': self.show_eraser,
                    },
                    {
                        'name': 'size', 'type': 'int',
                        'value': self.eraser_size,
                        'visible': False,
                    },
                    {
                        'name': 'enable', 'type': 'bool',
                        'value': self.enable_eraser,
                    },
                    {
                        'name': 'reset', 'type': 'action',
                    },
                ],
                'expanded': False,
            },
        ]
        self.calib_mask_params = Parameter.create(
            name='calib/mask parameters',
            type='group',
            children=calib_mask_params,
        )
        self.calibTree.setParameters(
            self.calib_mask_params, showTop=False)
        # hit finder parameter tree
        hit_finder_params = [
            {
                'name': 'hit finding on', 'type': 'bool',
                'value': self.hit_finding_on
            },
            {
                'name': 'mask on', 'type': 'bool', 'value': self.mask_on,
            },
            {
                'name': 'max peaks', 'type': 'int',
                'value': self.max_peaks,
            },
            {
                'name': 'min peaks', 'type': 'int',
                'value': self.min_peaks,
            },
            {
                'name': 'min pixels', 'type': 'int',
                'value': self.min_pixels
            },
            {
                'name': 'max pixels', 'type': 'int',
                'value': self.max_pixels
            },
            {
                'name': 'hit finder', 'type': 'list',
                'values': self.hit_finders, 'value': self.hit_finder,
            },
            {
                'name': 'poisson model', 'type': 'group', 'children': [
                    {
                        'name': 'adu per photon', 'type': 'int',
                        'value': self.adu_per_photon,
                    },
                    {
                        'name': 'epsilon', 'type': 'float',
                        'value': self.epsilon,
                    },
                    {
                        'name': 'bin size', 'type': 'int',
                        'value': self.bin_size,
                    },
                ],
                'visible': False,
                'expanded': False,
            },
            {
                'name': 'snr model', 'type': 'group', 'children': [
                    {
                        'name': 'gaussian filter sigma', 'type': 'float',
                        'value': self.gaussian_sigma,
                    },
                    {
                        'name': 'min gradient', 'type': 'float',
                        'value': self.min_gradient,
                    },
                    {
                        'name': 'min distance', 'type': 'int',
                        'value': self.min_distance
                    },
                    {
                        'name': 'clean flat peaks', 'type': 'bool',
                        'value': self.merge_flat_peaks
                    },
                    {
                        'name': 'crop size', 'type': 'int',
                        'value': self.crop_size,
                        'visible': False,
                    },
                    {
                        'name': 'peak refine mode', 'type': 'list',
                        'values': self.peak_refine_mode_list,
                        'value': self.peak_refine_mode,
                    },
                    {
                        'name': 'min snr', 'type': 'float',
                        'value': self.min_snr
                    },
                    {
                        'name': 'snr mode', 'type': 'list',
                        'values': self.snr_mode_list,
                        'value': self.snr_mode,
                    },
                    {
                        'name': 'simple', 'type': 'group', 'children': [
                            {
                                'name': 'background ratio', 'type': 'float',
                                'value': self.bg_ratio,
                            },
                            {
                                'name': 'signal ratio', 'type': 'float',
                                'value': self.sig_ratio,
                            },
                        ],
                        'visible': True if self.snr_mode == 'simple' else False
                    },
                    {
                        'name': 'rings', 'type': 'group', 'children': [
                            {
                                'name': 'signal radius', 'type': 'int',
                                'value': self.sig_radius,
                            },
                            {
                                'name': 'background inner radius',
                                'type': 'int',
                                'value': self.bg_inner_radius,
                            },
                            {
                                'name': 'background outer radius',
                                'type': 'int',
                                'value': self.bg_outer_radius,
                            },
                        ],
                        'visible': True if self.snr_mode == 'rings' else False
                    },
                    {
                        'name': 'adaptive', 'type': 'group', 'children': [
                            {
                                'name': 'background ratio', 'type': 'float',
                                'value': self.bg_ratio,
                            },
                            {
                                'name': 'signal threshold', 'type': 'float',
                                'value': self.sig_thres,
                            },
                        ],
                        'visible': True if self.snr_mode == 'adaptive'
                        else False
                    },

                ],
                'expanded': False
            },
            {
                'name': 'display', 'type': 'group', 'children': [
                    {
                        'name': 'show raw peaks', 'type': 'bool',
                        'value': self.show_raw_peaks,
                    },
                    {
                        'name': 'show valid peaks', 'type': 'bool',
                        'value': self.show_valid_peaks,
                    },
                    {
                        'name': 'show opt peaks', 'type': 'bool',
                        'value': self.show_opt_peaks,
                    },
                ],
                'expanded': False,
            }
        ]
        self.hit_finder_params = Parameter.create(
            name='hit finder parameters',
            type='group',
            children=hit_finder_params,
        )
        self.hitFinderTree.setParameters(
            self.hit_finder_params, showTop=False
        )
        # signal/slots
        # menu bar action
        self.actionAdd_File.triggered.connect(self.add_file)
        self.actionLoad_Hit_Finding_Conf.triggered.connect(self.load_hit_conf)
        self.actionLoad_Geometry.triggered.connect(self.load_geom_file)
        self.actionSave_Hit_Finding_Conf.triggered.connect(self.save_hit_conf)
        self.actionSave_Mask.triggered.connect(self.save_mask)
        self.actionCreate_Proxy_Data_LCLS.triggered.connect(
            self.show_data_proxy_dialog
        )
        self.actionSettings.triggered.connect(self.show_settings)
        self.actionShow_Calib_Mask_View.triggered.connect(
            self.show_or_hide_mask_view)
        self.actionShow_Debug_View.triggered.connect(
            self.show_or_hide_debug_view)
        self.actionHide_File_List.triggered.connect(
            self.show_or_hide_file_list)
        self.actionShow_Inspector.triggered.connect(self.show_inspector)
        self.actionPeak_Table.triggered.connect(self.show_peak_table)
        self.actionPowder_Fit.triggered.connect(self.show_powder_win)
        self.actionJob_Table.triggered.connect(self.show_job_win)
        self.actionStats_Viewer.triggered.connect(
            partial(self.show_stats_win, job=None, tag=None))
        # job table
        self.job_win.view_stats.connect(self.show_stats_win)
        # peak table
        self.peak_table.peak_table.cellDoubleClicked.connect(
            self.zoom_in_on_peak
        )
        # mean dialog
        self.mean_diag.datasetComboBox.currentIndexChanged.connect(
            self.update_mean_diag_nframe)
        self.mean_diag.meanButtonBox.clicked.connect(self.calc_mean_std)
        # proxy dialog
        self.proxy_diag.buttonBox.clicked.connect(self.create_proxy_data)
        # powder dialog
        self.powder_diag.datasetComboBox.currentIndexChanged.connect(
            self.update_powder_diag_nframe)
        self.powder_diag.submitButton.clicked.connect(self.gen_powder)
        # file list
        self.fileList.itemDoubleClicked.connect(self.load_file)
        self.fileList.customContextMenuRequested.connect(
            self.show_file_list_menu)
        self.addFileLine.returnPressed.connect(self.add_files)
        # image viewers
        self.rawView.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=1))
        self.maskView.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=2))
        self.debugView.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=3))
        # viewer tree
        self.viewer_params.param(
            'current frame'
        ).sigValueChanged.connect(self.change_frame)
        self.viewer_params.param(
            'show center'
        ).sigValueChanged.connect(self.change_show_center)
        self.viewer_params.param(
            'apply geom'
        ).sigValueChanged.connect(self.change_apply_geom)
        self.viewer_params.param(
            'display', 'auto range'
        ).sigValueChanged.connect(self.change_auto_range)
        self.viewer_params.param(
            'display', 'auto level'
        ).sigValueChanged.connect(self.change_auto_level)
        self.viewer_params.param(
            'display', 'auto hist range'
        ).sigValueChanged.connect(self.change_auto_histogram_range)
        # calib/mask
        self.calib_mask_params.param(
            'threshold'
        ).sigValueChanged.connect(self.change_mask_thres)
        self.calib_mask_params.param(
            'center x'
        ).sigValueChanged.connect(self.change_center_x)
        self.calib_mask_params.param(
            'center y'
        ).sigValueChanged.connect(self.change_center_y)
        self.calib_mask_params.param(
            'radii of rings'
        ).sigValueChanged.connect(self.change_ring_radii)
        self.calib_mask_params.param(
            'morphology', 'erosion1 size'
        ).sigValueChanged.connect(self.change_erosion1_size)
        self.calib_mask_params.param(
            'morphology', 'dilation size'
        ).sigValueChanged.connect(self.change_dilation_size)
        self.calib_mask_params.param(
            'morphology', 'erosion2 size'
        ).sigValueChanged.connect(self.change_erosion2_size)
        self.calib_mask_params.param(
            'eraser', 'show'
        ).sigValueChanged.connect(self.change_show_eraser)
        self.calib_mask_params.param(
            'eraser', 'enable'
        ).sigValueChanged.connect(self.change_enable_eraser)
        self.calib_mask_params.param(
            'eraser', 'reset'
        ).sigActivated.connect(self.reset_eraser)
        self.eraser_item.sigRegionChanged.connect(self.change_eraser)
        # hit finder
        self.hit_finder_params.param(
            'mask on'
        ).sigValueChanged.connect(self.change_mask_on)
        self.hit_finder_params.param(
            'hit finding on'
        ).sigValueChanged.connect(self.change_hit_finding_on)
        self.hit_finder_params.param(
            'max peaks'
        ).sigValueChanged.connect(self.change_max_peaks)
        self.hit_finder_params.param(
            'min pixels'
        ).sigValueChanged.connect(self.change_min_peaks)
        self.hit_finder_params.param(
            'min pixels'
        ).sigValueChanged.connect(self.change_min_pixels)
        self.hit_finder_params.param(
            'max pixels'
        ).sigValueChanged.connect(self.change_max_pixels)
        self.hit_finder_params.param(
            'hit finder'
        ).sigValueChanged.connect(self.change_hit_finder)
        self.hit_finder_params.param(
            'poisson model', 'adu per photon'
        ).sigValueChanged.connect(self.change_adu_per_photon)
        self.hit_finder_params.param(
            'poisson model', 'epsilon'
        ).sigValueChanged.connect(self.change_epsilon)
        self.hit_finder_params.param(
            'poisson model', 'bin size'
        ).sigValueChanged.connect(self.change_bin_size)
        self.hit_finder_params.param(
            'snr model', 'gaussian filter sigma'
        ).sigValueChanged.connect(self.change_gaussian_sigma)
        self.hit_finder_params.param(
            'snr model', 'min gradient'
        ).sigValueChanged.connect(self.change_min_gradient)
        self.hit_finder_params.param(
            'snr model', 'min distance'
        ).sigValueChanged.connect(self.change_min_distance)
        self.hit_finder_params.param(
            'snr model', 'clean flat peaks'
        ).sigValueChanged.connect(self.change_clean_flat_peaks)
        self.hit_finder_params.param(
            'snr model', 'crop size'
        ).sigValueChanged.connect(self.change_crop_size)
        self.hit_finder_params.param(
            'snr model', 'peak refine mode'
        ).sigValueChanged.connect(self.change_peak_refine_mode)
        self.hit_finder_params.param(
            'snr model', 'min snr'
        ).sigValueChanged.connect(self.change_min_snr)
        self.hit_finder_params.param(
            'snr model', 'snr mode'
        ).sigValueChanged.connect(self.change_snr_mode)
        self.hit_finder_params.param(
            'display', 'show raw peaks'
        ).sigValueChanged.connect(self.change_show_raw_peaks)
        self.hit_finder_params.param(
            'display', 'show valid peaks'
        ).sigValueChanged.connect(self.change_show_valid_peaks)
        self.hit_finder_params.param(
            'display', 'show opt peaks'
        ).sigValueChanged.connect(self.change_show_opt_peaks)

# menu bar related methods
    @pyqtSlot()
    def add_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File",
            '.', "Data (*.h5 *.cxi *.npy *.npz *.lcls)"
        )
        if len(path) == 0:
            return
        self.maybe_add_file(path)

    @pyqtSlot()
    def load_hit_conf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Hit Finding Conf File",
            '.', "Yaml Files (*.yml)"
        )
        if len(path) == 0:
            return
        self.add_info('Load hit configuration from %s' % path)

        with open(path, 'r') as f:
            conf_dict = yaml.load(f)
        self.dataset = conf_dict.get('dataset', None)
        self.mask_on = conf_dict.get('mask on', False)
        self.mask_file = conf_dict.get('mask file', None)
        if self.mask_on and self.mask_file is not None:
            self.mask = util.read_image(self.mask_file)['image']
        self.center = np.array(conf_dict.get('center', [0, 0]))
        self.hit_finder = conf_dict.get('hit finder', 'snr model')
        self.max_peaks = conf_dict.get('max peaks', 500)
        self.min_peaks = conf_dict.get('min peaks', 20)
        self.adu_per_photon = conf_dict.get('adu per photon', 20)
        self.epsilon = conf_dict.get('epsilon', 1e-5)
        self.bin_size = conf_dict.get('bin size', 4)
        self.gaussian_sigma = conf_dict.get('gaussian filter sigma', 1.)
        self.min_gradient = conf_dict.get('min gradient', 10)
        self.min_distance = conf_dict.get('min distance', 10)
        self.merge_flat_peaks = conf_dict.get('merge flat peaks', False)
        self.crop_size = conf_dict.get('crop size', 7)
        self.peak_refine_mode = conf_dict.get('peak refine mode', 'gradient')
        self.min_snr = conf_dict.get('min snr', 6)
        self.min_pixels = conf_dict.get('min pixels', 2)
        self.max_pixels = conf_dict.get('max pixels', 10)
        self.snr_mode = conf_dict.get('snr mode', 'adaptive')
        self.bg_ratio = conf_dict.get('background ratio', 0.7)
        self.sig_ratio = conf_dict.get('signal ratio', 0.2)
        self.bg_inner_radius = conf_dict.get('background inner radius', 2)
        self.bg_outer_radius = conf_dict.get('background outer radius', 3)
        self.sig_thres = conf_dict.get('signal threshold', 5.)
        # update status and parameters for display
        self.update_file_info()
        self.calib_mask_params.param('center x').setValue(self.center[0])
        self.calib_mask_params.param('center y').setValue(self.center[1])
        self.hit_finder_params.param('mask on').setValue(self.mask_on)
        self.hit_finder_params.param('hit finder').setValue(self.hit_finder)
        self.hit_finder_params.param(
            'max peaks'
        ).setValue(self.max_peaks)
        self.hit_finder_params.param(
            'min peaks'
        ).setValue(self.min_peaks)
        self.hit_finder_params.param(
            'min pixels'
        ).setValue(self.min_pixels)
        self.hit_finder_params.param(
            'max pixels'
        ).setValue(self.max_pixels)
        self.hit_finder_params.param(
            'poisson model', 'adu per photon'
        ).setValue(self.adu_per_photon)
        self.hit_finder_params.param(
            'poisson model', 'epsilon'
        ).setValue(self.epsilon)
        self.hit_finder_params.param(
            'poisson model', 'bin size'
        ).setValue(self.bin_size)
        self.hit_finder_params.param(
            'snr model', 'gaussian filter sigma'
        ).setValue(self.gaussian_sigma)
        self.hit_finder_params.param(
            'snr model', 'min gradient'
        ).setValue(self.min_gradient)
        self.hit_finder_params.param(
            'snr model', 'min distance'
        ).setValue(self.min_distance)
        self.hit_finder_params.param(
            'snr model', 'clean flat peaks'
        ).setValue(self.merge_flat_peaks)
        self.hit_finder_params.param(
            'snr model', 'crop size'
        ).setValue(self.crop_size)
        self.hit_finder_params.param(
            'snr model', 'peak refine mode'
        ).setValue(self.peak_refine_mode)
        self.hit_finder_params.param(
            'snr model', 'min snr'
        ).setValue(self.min_snr)
        self.hit_finder_params.param(
            'snr model', 'snr mode'
        ).setValue(self.snr_mode)
        self.hit_finder_params.param(
            'snr model', 'rings', 'signal radius'
        ).setValue(self.sig_radius)
        self.hit_finder_params.param(
           'snr model', 'rings', 'background inner radius'
        ).setValue(self.bg_inner_radius)
        self.hit_finder_params.param(
            'snr model', 'rings', 'background outer radius'
        ).setValue(self.bg_outer_radius)
        self.hit_finder_params.param(
            'snr model', 'simple', 'background ratio'
        ).setValue(self.bg_ratio)
        self.hit_finder_params.param(
            'snr model', 'adaptive', 'background ratio'
        ).setValue(self.bg_ratio)
        self.hit_finder_params.param(
            'snr model', 'simple', 'signal ratio'
        ).setValue(self.sig_ratio)
        self.hit_finder_params.param(
            'snr model', 'adaptive', 'signal threshold'
        ).setValue(self.sig_thres)

    @pyqtSlot()
    def load_geom_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Geom File",
            '.', "Geometry (*.geom *.h5 *. *.npz *.data)"
        )
        if len(path) == 0:
            return
        self.geom_file = path
        self.geom = geometry.Geometry(self.geom_file, 110)  # TODO
        self.add_info('Load geometry file: %s' % self.geom_file)

    @pyqtSlot()
    def save_hit_conf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Hit Finding Conf File",
            'conf/hit_finding', "Yaml Files (*.yml)"
        )
        if len(path) == 0:
            return
        conf_dict = {
            'dataset': self.dataset,
            'mask file': self.mask_file,
            'center': self.center.tolist(),
            'mask on': self.mask_on,
            'hit finder': self.hit_finder,
            'max peaks': self.max_peaks,
            'min peaks': self.min_peaks,
            'adu per photon': self.adu_per_photon,
            'epsilon': self.epsilon,
            'bin size': self.bin_size,
            'gaussian filter sigma': self.gaussian_sigma,
            'min gradient': self.min_gradient,
            'min distance': self.min_distance,
            'merge flat peaks': self.merge_flat_peaks,
            'crop size': self.crop_size,
            'peak refine mode': self.peak_refine_mode,
            'min snr': self.min_snr,
            'min pixels': self.min_pixels,
            'max pixels': self.max_pixels,
            'snr mode': self.snr_mode,
            'background ratio': self.bg_ratio,
            'signal ratio': self.sig_ratio,
            'signal radius': self.sig_radius,
            'background inner radius': self.bg_inner_radius,
            'background outer radius': self.bg_outer_radius,
            'signal threshold': self.sig_thres,
        }
        with open(path, 'w') as f:
            yaml.dump(conf_dict, f, default_flow_style=False)
        self.add_info('Save hit configuration to %s' % path)

    @pyqtSlot()
    def show_or_hide_mask_view(self):
        self.show_mask_view = not self.show_mask_view
        if self.show_mask_view:
            self.actionShow_Calib_Mask_View.setText('Hide Calib/Mask View')
            self.maskView.show()
        else:
            self.actionShow_Calib_Mask_View.setText('Show Calib/Mask View')
            self.maskView.hide()
        self.update_display()

    @pyqtSlot()
    def show_or_hide_debug_view(self):
        self.show_debug_view = not self.show_debug_view
        if self.show_debug_view:
            self.actionShow_Debug_View.setText('Hide Debug View')
            self.debugView.show()
        else:
            self.actionShow_Debug_View.setText('Show Debug View')
            self.debugView.hide()
        self.update_display()

    @pyqtSlot()
    def show_or_hide_file_list(self):
        self.show_file_list = not self.show_file_list
        if self.show_file_list:
            self.actionHide_File_List.setText('Hide File List')
            self.fileListFrame.show()
        else:
            self.actionHide_File_List.setText('Show File List')
            self.fileListFrame.hide()

    @pyqtSlot()
    def save_mask(self):
        if self.mask_image is None:
            self.add_info('No mask image available')
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save mask to", 'mask', "npy file(*.npy)"
            )
            if len(path) == 0:
                return
            np.save(path, self.mask_image)
            self.add_info('Mask saved to %s' % path)

    @pyqtSlot()
    def show_data_proxy_dialog(self):
        self.proxy_diag.show()

    @pyqtSlot()
    def show_settings(self):
        self.settings.load_settings()
        if self.setting_diag.exec_() == QDialog.Accepted:
            self.settings.save_settings()

    @pyqtSlot()
    def show_inspector(self):
        self.inspector.show()

    @pyqtSlot()
    def show_peak_table(self):
        self.peak_table.show()

    @pyqtSlot()
    def show_powder_win(self):
        self.powder_win.update_settings(self.settings)
        self.powder_win.update_peaks_view()
        self.powder_win.showMaximized()

    @pyqtSlot()
    def show_job_win(self):
        self.job_win.update_settings(self.settings)
        self.job_win.showMaximized()
        self.job_win.start()

# calib/mask related methods
    @pyqtSlot(object, object)
    def change_show_center(self, _, show_center):
        self.show_center = show_center
        self.update_display()

    @pyqtSlot(object, object)
    def change_apply_geom(self, _, apply_geom):
        self.apply_geom = apply_geom
        self.update_file_info()
        self.update_display()

    @pyqtSlot(object, object)
    def change_mask_thres(self, _, thres):
        self.mask_thres = thres
        self.mask_image = (self.raw_image > thres).astype(np.int)
        self.change_image()
        self.update_display()

    @pyqtSlot(object, object)
    def change_center_x(self, _, x):
        self.center[0] = x
        self.update_display()

    @pyqtSlot(object, object)
    def change_center_y(self, _, y):
        self.center[1] = y
        self.update_display()

    @pyqtSlot(object, object)
    def change_ring_radii(self, _, radii_str):
        if len(radii_str) > 0:
            self.ring_radii = np.array(list(map(float, radii_str.split(','))))
        else:
            self.ring_radii = []
        self.update_display()

    @pyqtSlot(object, object)
    def change_erosion1_size(self, _, size):
        self.erosion1_size = size
        self.change_image()
        self.update_display()

    @pyqtSlot(object, object)
    def change_dilation_size(self, _, size):
        self.dilation_size = size
        self.change_image()
        self.update_display()

    @pyqtSlot(object, object)
    def change_erosion2_size(self, _, size):
        self.erosion2_size = size
        self.change_image()
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_eraser(self, _, show):
        if show:
            self.eraser_item.show()
            self.show_eraser = True
        else:
            self.eraser_item.hide()
            self.show_eraser = False

    @pyqtSlot(object, object)
    def change_enable_eraser(self, _, enable):
        if enable:
            self.enable_eraser = True
            self.change_eraser(self.eraser_item) # make eraser mask immediately
        else:
            self.enable_eraser = False

    @pyqtSlot()
    def reset_eraser(self):
        self.eraser_mask = np.ones_like(self.raw_image, dtype=np.int)
        self.change_image()
        self.update_display()

    @pyqtSlot(object)
    def change_eraser(self, eraser):
        if self.enable_eraser and self.raw_image is not None:
            pos = eraser.pos()
            radius = eraser.size()[0] / 2
            center = (pos[0] + radius, pos[1] + radius)
            mask = util.make_circle_mask(self.raw_image.shape, center, radius)
            if self.eraser_mask is None:
                self.eraser_mask = mask.copy()
            else:
                self.eraser_mask *= mask
            self.change_image()
            self.update_display()

# hit finder related methods
    @pyqtSlot(object, object)
    def change_mask_on(self, _, mask_on):
        self.mask_on = mask_on
        if self.mask_on and self.mask_file is not None:
            mask = util.read_image(self.mask_file)['image']
            self.mask = mask
        else:
            self.mask = None
        self.update_display()

    @pyqtSlot(object, object)
    def change_hit_finding_on(self, _, hit_finding_on):
        self.hit_finding_on = hit_finding_on
        self.update_display()

    @pyqtSlot(object, object)
    def change_max_peaks(self, _, max_peaks):
        self.max_peaks = max_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_peaks(self, _, min_peaks):
        self.min_peaks = min_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_pixels(self, _, min_pixels):
        self.min_pixels = min_pixels
        self.update_display()

    @pyqtSlot(object, object)
    def change_max_pixels(self, _, max_pixels):
        self.max_pixels = max_pixels
        self.update_display()

    @pyqtSlot(object, object)
    def change_gaussian_sigma(self, _, gaussian_sigma):
        self.gaussian_sigma = gaussian_sigma
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_gradient(self, _, min_gradient):
        self.min_gradient = min_gradient
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_distance(self, _, min_distance):
        self.min_distance = min_distance
        self.update_display()

    @pyqtSlot(object, object)
    def change_clean_flat_peaks(self, _, clean_flat_peaks):
        self.merge_flat_peaks = clean_flat_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_crop_size(self, _, crop_size):
        self.crop_size = crop_size
        self.update_display()

    @pyqtSlot(object, object)
    def change_peak_refine_mode(self, _, peak_refine_mode):
        self.peak_refine_mode = peak_refine_mode
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_snr(self, _, min_snr):
        self.min_snr = min_snr
        self.update_display()

    @pyqtSlot(object, object)
    def change_hit_finder(self, _, hit_finder):
        self.hit_finder = hit_finder
        self.add_info('Using %s hit finder' % hit_finder)
        for hit_finder in self.hit_finders:
            self.hit_finder_params.param(hit_finder).hide()
        self.hit_finder_params.param(self.hit_finder).show()
        self.hitFinderTree.setParameters(
            self.hit_finder_params, showTop=False
        )
        self.update_display()

    @pyqtSlot(object, object)
    def change_adu_per_photon(self, _, adu_per_photon):
        self.adu_per_photon = adu_per_photon
        self.update_display()

    @pyqtSlot(object, object)
    def change_epsilon(self, _, epsilon):
        self.epsilon = epsilon
        self.update_display()

    @pyqtSlot(object, object)
    def change_bin_size(self, _, bin_size):
        self.bin_size = bin_size
        self.update_display()

    @pyqtSlot(object, object)
    def change_snr_mode(self, _, snr_mode):
        self.add_info('Using %s snr mode' % snr_mode)
        self.snr_mode = snr_mode
        for snr_mode in self.snr_mode_list:
            self.hit_finder_params.param('snr model', snr_mode).hide()
        self.hit_finder_params.param('snr model', self.snr_mode).show()
        self.hitFinderTree.setParameters(
            self.hit_finder_params, showTop=False
        )
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_raw_peaks(self, _, show):
        self.show_raw_peaks = show
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_valid_peaks(self, _, show):
        self.show_valid_peaks = show
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_opt_peaks(self, _, show):
        self.show_opt_peaks = show
        self.update_display()

# image viewers
    @pyqtSlot(object)
    def mouse_moved(self, pos, flag=None):
        if self.path is None:
            return
        if flag == 1:  # in raw view
            mouse_point = self.rawView.view.mapToView(pos)
        elif flag == 2:  # in mask view
            mouse_point = self.maskView.view.mapToView(pos)
        elif flag == 3:  # debug view
            mouse_point = self.debugView.view.mapToView(pos)
        else:
            return
        x, y = int(mouse_point.x()), int(mouse_point.y())
        if 0 <= x < self.raw_image.shape[0] and 0 <= y < self.raw_image.shape[1]:
            message = 'x:%d y:%d, I(raw): %.2E;' % (x, y, self.raw_image[x, y])
            if self.show_mask_view and self.mask_image is not None:
                message += 'I(calib/mask): %.2E' % self.mask_image[x, y]
            if self.show_debug_view and self.debug_image is not None:
                message += 'I(debug): %.2E' % self.debug_image[x, y]
            self.statusbar.showMessage(message, 5000)
        else:
            return
        if self.inspector.isVisible():  # show data inspector
            # out of bound check
            if x - 3 < 0 or x + 4 > self.raw_image.shape[0]:
                return
            elif y - 3 < 0 or y + 4 > self.raw_image.shape[1]:
                return
            # calculate snr
            pos = np.reshape((x, y), (-1, 2))
            if self.hit_finder == 'poisson model':
                snr_mode = 'threshold'
            else:
                snr_mode = self.snr_mode
            snr_info = util.calc_snr(
                self.raw_image, pos,
                mode=snr_mode,
                signal_radius=self.sig_radius,
                bg_inner_radius=self.bg_inner_radius,
                bg_outer_radius=self.bg_outer_radius,
                crop_size=self.crop_size,
                bg_ratio=self.bg_ratio,
                signal_ratio=self.sig_ratio,
                signal_thres=self.sig_thres,
                thres_map=self.thres_map,
                label_pixels=True,
            )
            self.inspector.snrLabel.setText('SNR@(%d, %d):' % (x, y))
            self.inspector.snrValue.setText(
                '%.1f(sig %.1f, bg %.1f, noise %.1f)' %
                (snr_info['snr'][0],
                 snr_info['signal values'][0],
                 snr_info['background values'][0],
                 snr_info['noise values'][0]))
            # set table values
            if self.maskView.isVisible():
                self.inspector.maskButton.setEnabled(True)
            else:
                self.inspector.maskButton.setEnabled(False)
            if self.debugView.isVisible():
                self.inspector.debugButton.setEnabled(True)
            else:
                self.inspector.debugButton.setEnabled(False)
            sig_pixels = (snr_info['signal pixels'] - pos + 3).tolist()
            bg_pixels = (snr_info['background pixels'] - pos + 3).tolist()
            if self.inspector.rawButton.isChecked():
                inspector_image = self.raw_image
            elif self.inspector.maskButton.isChecked():
                inspector_image = self.mask_image
            else:
                inspector_image = self.debug_image
            for i in range(7):
                for j in range(7):
                    v1 = inspector_image[x + i - 3, y + j - 3]
                    item = QTableWidgetItem('%d' % v1)
                    item.setTextAlignment(Qt.AlignCenter)
                    if [i, j] in sig_pixels:
                        item.setBackground(QtGui.QColor(178, 247, 143))
                    elif [i, j] in bg_pixels:
                        item.setBackground(QtGui.QColor(165, 173, 186))
                    else:
                        item.setBackground(QtGui.QColor(255, 255, 255))
                    self.inspector.data_table.setItem(j, i, item)

    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > self.total_frames - 1:
            frame = self.total_frames - 1
        self.curr_frame = frame
        self.viewer_params.param(
            'current frame'
        ).setValue(self.curr_frame)
        self.change_image()
        self.update_display()

    @pyqtSlot(object, object)
    def change_auto_range(self, _, auto_range):
        self.auto_range = auto_range
        self.update_display()

    @pyqtSlot(object, object)
    def change_auto_level(self, _, auto_level):
        self.auto_level = auto_level
        self.update_display()

    @pyqtSlot(object, object)
    def change_auto_histogram_range(self, _, auto_histogram_range):
        self.auto_histogram_range = auto_histogram_range
        self.update_display()

    def change_image(self):
        if self.path is None:
            return
        raw_image = util.read_image(
            self.path, frame=self.curr_frame,
            h5_obj=self.h5_obj, dataset=self.dataset,
            lcls_data=self.lcls_data,
        )['image']
        if raw_image is None:  # skip None image
            self.add_info('NoneType image found', info_type='WARNING')
            return
        self.raw_image = raw_image.astype(np.float)
        self.mask_image = util.make_simple_mask(
            self.raw_image, self.mask_thres, erosion1=self.erosion1_size,
            dilation=self.dilation_size, erosion2=self.erosion2_size
        ).astype(np.float)
        if self.eraser_mask is not None and (
                self.eraser_mask.shape == self.raw_image.shape):
            self.mask_image *= self.eraser_mask.astype(np.float)

    def update_display(self):
        if self.raw_image is None:
            return
        # apply mask
        mask = np.ones_like(self.raw_image, dtype=np.float)
        if self.mask_on and self.mask is not None:
            mask *= self.mask
        if self.mask_image is not None:
            mask *= self.mask_image
        raw_image = self.raw_image * mask
        # apply geom
        if self.apply_geom and self.geom is not None:
            raw_image = self.geom.raw2assembled(raw_image)
            mask = self.geom.raw2assembled(mask)
        self.rawView.setImage(
            raw_image, autoRange=self.auto_range,
            autoLevels=self.auto_level,
            autoHistogramRange=self.auto_histogram_range)
        if self.maskView.isVisible():
            self.maskView.setImage(
                self.mask_image, autoRange=self.auto_range,
                autoLevels=self.auto_level,
                autoHistogramRange=self.auto_histogram_range)
        # clear all plot items
        self.raw_peak_item.clear()
        self.valid_peak_item.clear()
        self.opt_peak_item.clear()
        self.strong_peak_item.clear()
        self.raw_center_item.clear()
        self.raw_ring_item.clear()
        self.mask_center_item.clear()
        self.mask_ring_item.clear()

        if self.hit_finding_on:
            peaks_dict = util.find_peaks(
                raw_image, self.center,
                adu_per_photon=self.adu_per_photon,
                epsilon=self.epsilon,
                bin_size=self.bin_size,
                mask=mask,
                hit_finder=self.hit_finder,
                gaussian_sigma=self.gaussian_sigma,
                min_gradient=self.min_gradient,
                min_distance=self.min_distance,
                merge_flat_peaks=self.merge_flat_peaks,
                max_peaks=self.max_peaks,
                min_snr=self.min_snr,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                refine_mode=self.peak_refine_mode,
                snr_mode=self.snr_mode,
                signal_radius=self.sig_radius,
                bg_inner_radius=self.bg_inner_radius,
                bg_outer_radius=self.bg_outer_radius,
                crop_size=self.crop_size,
                bg_ratio=self.bg_ratio,
                signal_ratio=self.sig_ratio,
                signal_thres=self.sig_thres,
                label_pixels=False,
            )
            raw_peaks = peaks_dict.get('raw', None)
            if raw_peaks is not None and self.show_raw_peaks:
                self.add_info('%d raw peaks found' % len(raw_peaks))
                self.raw_peak_item.setData(pos=raw_peaks + 0.5)
            valid_peaks = peaks_dict.get('valid', None)
            if valid_peaks is not None and self.show_valid_peaks:
                self.add_info(
                    '%d peaks remaining after mask cleaning'
                    % len(peaks_dict['valid'])
                )
                self.valid_peak_item.setData(pos=valid_peaks + 0.5)
            # refine peak position
            opt_peaks = peaks_dict.get('opt', None)
            if opt_peaks is not None and self.show_opt_peaks:
                self.opt_peak_item.setData(pos=opt_peaks + 0.5)
            # filtering weak peak
            strong_peaks = peaks_dict.get('strong', None)
            if strong_peaks is not None and self.show_strong_peaks:
                self.add_info('%d strong peaks' % (len(strong_peaks)))
                if len(strong_peaks) > 0:
                    self.strong_peak_item.setData(pos=strong_peaks + 0.5)
            self.strong_peaks = strong_peaks

            debug_image = peaks_dict.get('thres_map', None)
            if debug_image is not None and self.debugView.isVisible():
                self.debugView.setImage(debug_image)
                self.debug_image = debug_image
            self.thres_map = peaks_dict.get('thres_map', None)

            # update peak table if visible
            self.peak_info = peaks_dict.get('info', None)
            if self.peak_info is not None:
                self.update_peak_table(self.peak_info)

        # center item
        if self.show_center:
            self.raw_center_item.setData(pos=self.center.reshape(1, 2) + 0.5)
            if self.maskView.isVisible():
                self.mask_center_item.setData(
                    pos=self.center.reshape(1, 2) + 0.5)
        # ring item
        if len(self.ring_radii) > 0:
            centers = np.repeat(
                self.center.reshape(1, 2), len(self.ring_radii), axis=0)
            self.raw_ring_item.setData(
                pos=centers + 0.5, size=self.ring_radii * 2.)
            if self.maskView.isVisible():
                self.mask_ring_item.setData(
                    pos=centers + 0.5, size=self.ring_radii * 2.)

    def update_peak_table(self, peak_info):
        table = self.peak_table.peak_table
        nb_peaks = peak_info['snr'].size
        data = []
        for i in range(nb_peaks):
            data.append((
                i,
                peak_info['snr'][i],
                peak_info['total intensity'][i],
                peak_info['signal values'][i],
                peak_info['background values'][i],
                peak_info['noise values'][i],
                peak_info['signal pixel num'][i],
                peak_info['background pixel num'][i],
                peak_info['radius'][i],
            ))
        data = np.array(data, dtype=[
            ('id', int),
            ('snr', float),
            ('total intensity', float),
            ('signal', float),
            ('background', float),
            ('noise', float),
            ('signal pixels', int),
            ('background pixels', int),
            ('radius', float),
        ])
        table.setData(data)
        # format cells
        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                item = table.item(i, j)
                item.setTextAlignment(Qt.AlignCenter)

# job table
    @pyqtSlot(str, str)
    def show_stats_win(self, job, tag):
        if job is not None and tag is not None:
            stats_file = os.path.join(
                '.', 'cxi_hit', job, tag, '%s.npy' % job)
            self.stats_win.statsFile.setText(stats_file)
            self.stats_win.load_stats(stats_file)
        self.stats_win.show()

# peak table
    @pyqtSlot(int, int)
    def zoom_in_on_peak(self, row, _):
        table = self.peak_table.peak_table
        peak_id = int(table.item(row, 0).text())
        x, y = self.strong_peaks[peak_id]
        # raw view
        self.rawView.view.setRange(
            xRange=(x-20, x+20), yRange=(y-20, y+20)
        )
        if self.peak_info is not None:
            bg = self.peak_info['background values'][peak_id]
            noise = self.peak_info['noise values'][peak_id]
            self.rawView.setLevels(bg, bg + self.sig_thres * noise)
        # calib view
        if self.maskView.isVisible():
            self.maskView.view.setRange(
                xRange=(x - 20, x + 20), yRange=(y - 20, y + 20)
            )
        # debug view
        if self.debugView.isVisible():
            self.debugView.view.setRange(
                xRange=(x - 20, x + 20), yRange=(y - 20, y + 20)
            )

# mean dialog
    @pyqtSlot()
    def calc_mean_std(self):
        if self.calc_mean_thread is not None:
            self.calc_mean_thread.terminate()
        self.add_info('Calculating mean/sigma...')
        self.mean_diag.meanButtonBox.button(
            QDialogButtonBox.Apply
        ).setEnabled(False)
        selected_items = self.fileList.selectedItems()
        files = []
        for item in selected_items:
            files.append(item.data(1))
        dataset = self.mean_diag.datasetComboBox.currentText()
        nb_frame = int(self.mean_diag.totalFrameLabel.text())
        max_frame = min(int(self.mean_diag.usedFrameBox.text()), nb_frame)
        prefix = self.mean_diag.prefixLine.text()
        output = os.path.join('mean', '%s.npz' % prefix)

        self.calc_mean_thread = MeanCalculatorThread(
            files=files, dataset=dataset, max_frame=max_frame, output=output
        )
        self.calc_mean_thread.update_progress.connect(
            self.update_progressbar
        )
        self.calc_mean_thread.finished.connect(
            partial(self.calc_mean_finished, output)
        )
        self.calc_mean_thread.start()

# data proxy dialog
    def create_proxy_data(self):
        exp_id = self.proxy_diag.expID.text()
        det_name = self.proxy_diag.detName.text()
        if len(exp_id) == 0:
            self.add_info('Please specify experiment to generate proxy data.')
            return
        if len(det_name) == 0:
            self.add_info('Please specify detector to generate proxy data.')
            return
        run_start = self.proxy_diag.runStart.value()
        run_end = self.proxy_diag.runEnd.value()
        clen_str = self.proxy_diag.clenStr.text()
        event_codes = self.proxy_diag.eventCodes.isChecked()
        flow_rate = self.proxy_diag.flowRate.text()
        pressure_str = self.proxy_diag.pressureStr.text()
        for run in range(run_start, run_end+1):
            with open('xtc_proxy/r%04d.lcls' % run, 'w') as f:
                f.write('exp: "%s"\n' % exp_id)
                f.write('det: "%s"\n' % det_name)
                if len(clen_str) > 0:
                    f.write('clen: "%s"\n' % clen_str)
                if event_codes:
                    f.write('evr: "evr1"\n')
                if len(flow_rate) > 0:
                    f.write('flow_rate: "%s"\n' % flow_rate)
                if len(pressure_str) > 0:
                    f.write('pressure: "%s"\n' % pressure_str)
            self.add_info('proxy data for run %d is created' % run)

    @pyqtSlot()
    def choose_dir(self, line_edit=None):
        if line_edit is not None:
            curr_dir = line_edit.text()
        else:
            curr_dir = ''
        dir_ = QFileDialog.getExistingDirectory(
            self, 'Choose directory', '.'
        )
        if len(dir_) == 0:
            return
        if line_edit is not None:
            line_edit.setText(dir_)

    @pyqtSlot(float)
    def update_progressbar(self, val):
        self.mean_diag.progressBar.setValue(val)

    @pyqtSlot()
    def calc_mean_finished(self, dest_file):
        self.add_info('Mean/sigma calculation done. File saved to %s.' % dest_file)

    @pyqtSlot(int)
    def update_mean_diag_nframe(self, curr_index):
        if curr_index == -1:
            return
        dataset = self.mean_diag.datasetComboBox.itemText(curr_index)
        selected_items = self.fileList.selectedItems()
        nb_frame = 0
        for item in selected_items:
            filepath = item.data(1)
            try:
                data_shape = util.get_data_shape(filepath)
                nb_frame += data_shape[dataset][0]
            except IOError:
                self.add_info('Failed to open %s' % filepath)
                pass
        self.mean_diag.totalFrameLabel.setText(str(nb_frame))

# powder dialog
    @pyqtSlot(int)
    def update_powder_diag_nframe(self, _):
        tag = self.powder_diag.datasetComboBox.currentText()
        if tag == '':
            return
        conf_file = 'conf/hit_finding/%s.yml' % tag
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)
        dataset = conf['dataset']
        selected_items = self.fileList.selectedItems()
        nb_frame = 0
        for item in selected_items:
            filepath = item.data(1)
            try:
                data_shape = util.get_data_shape(filepath)
                nb_frame += data_shape[dataset][0]
            except IOError:
                self.add_info('Failed to open %s' % filepath)
                pass
        self.powder_diag.totalFrameLabel.setText(str(nb_frame))

    @pyqtSlot()
    def gen_powder(self):
        selected_items = self.fileList.selectedItems()
        files = []
        for item in selected_items:
            files.append(item.data(1))
        powder_diag = self.powder_diag
        tag = powder_diag.datasetComboBox.currentText()
        if len(tag) == 0:
            self.add_info('No valid hit configuration.', info_type='WARNING')
            return
        conf_file = 'conf/hit_finding/%s.yml' % tag
        nb_frame = int(powder_diag.totalFrameLabel.text())
        max_frame = min(int(powder_diag.usedFrameBox.text()), nb_frame)
        if max_frame == 0:
            self.add_info('0 frame found.', info_type='WARNING')
        filename = powder_diag.prefixLine.text()
        output = os.path.join('powder', '%s.npz' % filename)
        self.powder_diag.submitButton.setEnabled(False)
        self.add_info('Submit powder generation job.')
        self.add_info('Powder calculation will take some time. Please wait '
                      'until the powder generation done!')
        self.gen_powder_thread = GenPowderThread(
            files, conf_file, self.settings,
            max_frame=max_frame,
            output=output,
        )
        self.gen_powder_thread.info.connect(self.add_info)
        # self.gen_powder_thread.finished.connect(
        #     self.gen_powder_finished
        # )
        self.gen_powder_thread.start()

    # @pyqtSlot()
    # def gen_powder_finished(self):
    #     self.add_info('Powder calculation will take some time. Please wait '
    #                   'until the powder file generated!')

# file list related methods
    @pyqtSlot()
    def add_files(self):
        path = self.addFileLine.text()
        files = glob(path)
        for f in files:
            self.maybe_add_file(os.path.abspath(f))

    @pyqtSlot(QPoint)
    def show_file_list_menu(self, pos):
        menu = QMenu()
        item = self.fileList.currentItem()
        if item is None:
            return
        path = item.data(1)
        ext = path.split('.')[-1]
        action_select_and_load_dataset = menu.addAction(
            'select and load dataset'
        )
        menu.addSeparator()
        action_set_as_mask = menu.addAction('set as mask')
        action_multiply_masks = menu.addAction('multiply masks')
        menu.addSeparator()
        action_calc_mean_std = menu.addAction('calculate mean/sigma')
        action_gen_powder = menu.addAction('generate peak powder')
        menu.addSeparator()
        action_del_file = menu.addAction('delete file(s)')
        action = menu.exec_(self.fileList.mapToGlobal(pos))
        if action == action_select_and_load_dataset:
            if ext == 'npy':
                self.path = path
                self.total_frames = 1
                self.curr_frame = 0
                self.dataset = 'None'
                self.add_info("Load %s" % path)
            elif ext == 'lcls':
                dataset = 'lcls-data'
                lcls_data = util.get_lcls_data(path)
                data_shape = util.get_data_shape(path)
                nb_frame = data_shape[dataset][0]
                self.path = path
                self.dataset = dataset
                self.total_frames = nb_frame
                self.lcls_data = lcls_data
                self.shape = data_shape[dataset][1:]
            elif ext in ('npz', 'h5', 'cxi'):
                data_shape = util.get_data_shape(path)
                dataset = self.select_dataset(path)
                if dataset is None:
                    return
                if len(data_shape[dataset]) == 3:
                    nb_frame = data_shape[dataset][0]
                    shape = data_shape[dataset][1:]
                else:
                    nb_frame = 1
                    shape = data_shape[dataset]
                if ext in ('cxi', 'h5'):
                    h5_obj = h5py.File(path, 'r')
                    self.h5_obj = h5_obj
                self.path = path
                self.dataset = dataset
                self.shape = shape
                self.total_frames = nb_frame
                if self.curr_frame >= self.total_frames:
                    self.curr_frame = self.total_frames - 1
                self.add_info("Load frame %d of %s-%s"
                              % (self.curr_frame, self.path, self.dataset))
            else:
                self.add_info('File type not supported: %s' % path)
            # update file info and display
            self.update_file_info()
            self.change_image()
            self.update_display()
        elif action == action_set_as_mask:
            self.mask_file = path
            mask = util.read_image(path)['image']
            total_pixels = mask.size
            valid_pixels = np.sum(mask)
            self.add_info('%d/%d(%.3f) valid pixels' %
                          (valid_pixels, total_pixels,
                           float(valid_pixels)/float(total_pixels)))
            self.mask = mask
            self.update_file_info()
            self.change_image()
            self.update_display()
        elif action == action_multiply_masks:
            items = self.fileList.selectedItems()
            mask_files = []
            for item in items:
                path = item.data(1)
                if path.split('.')[-1] == 'npy':
                    mask_files.append(path)
            if len(mask_files) == 0:
                return
            save_file, _ = QFileDialog.getSaveFileName(
                self, "Save mask", '.', "npy file(*.npy)"
            )
            if len(save_file) == 0:
                return
            mask = util.multiply_masks(mask_files)
            np.save(save_file, mask)
            self.add_info(
                'Making mask %s from %s' % (save_file, mask_files)
            )
        elif action == action_calc_mean_std:
            if ext == 'npy':
                self.add_info(
                    'Unsupported file type for mean calculation: %s' % ext
                )
                return  # ignore npy files
            combo_box = self.mean_diag.datasetComboBox
            combo_box.clear()
            data_shape = util.get_data_shape(path)
            for dataset, shape in data_shape.items():
                combo_box.addItem(dataset)
            self.mean_diag.progressBar.setValue(0)
            self.mean_diag.meanButtonBox.button(
                QDialogButtonBox.Apply
            ).setEnabled(True)
            self.mean_diag.exec_()
        elif action == action_gen_powder:
            confs = glob('conf/hit_finding/*.yml')
            self.powder_diag.datasetComboBox.clear()
            for conf in confs:
                tag = os.path.basename(conf).split('.')[0]
                self.powder_diag.datasetComboBox.addItem(tag)
            self.powder_diag.submitButton.setEnabled(True)
            self.powder_diag.exec_()
        elif action == action_del_file:
            items = self.fileList.selectedItems()
            for item in items:
                row = self.fileList.row(item)
                self.fileList.takeItem(row)
                self.all_files.remove(item.data(1))
                self.add_info('Remove %s' % item.data(1))

    @pyqtSlot('QListWidgetItem*')
    def load_file(self, item):
        path = item.data(1)
        self.add_info('Load %s' % path)
        self.load_data(path)
        self.update_file_info()
        self.change_image()
        self.update_display()

    def update_file_info(self):
        self.status_params.param('filepath').setValue(self.path)
        self.status_params.param('dataset').setValue(self.dataset)
        self.status_params.param('mask file').setValue(self.mask_file)
        self.status_params.param('geom file').setValue(self.geom_file)
        if self.apply_geom and self.geom is not None:
            shape_str = str(self.shape) + ' -> ' + str(self.geom.shape)
        else:
            shape_str = str(self.shape)
        self.status_params.param('shape').setValue(shape_str)
        self.status_params.param('total frame').setValue(self.total_frames)
        self.viewer_params.param('current frame').setValue(self.curr_frame)

    def load_data(self, path, dataset=None, frame=None):
        if dataset is None:
            dataset = self.dataset
        if frame is not None:
            self.curr_frame = frame
        ext = path.split('.')[-1]
        if ext == 'npy':
            self.path = path
            self.dataset = 'None'
            self.total_frames = 1
            self.shape = np.load(path).shape
        elif ext == 'npz':
            data_shape = util.get_data_shape(path)
            if dataset not in data_shape:
                dataset = self.select_dataset(path)
                if dataset is None:
                    return
            if len(data_shape[dataset]) == 3:
                nb_frame = data_shape[dataset][0]
            else:
                nb_frame = 1
            self.path = path
            self.dataset = dataset
            self.total_frames = nb_frame
            self.shape = np.load(path)[dataset].shape
        elif ext in ('h5', 'cxi'):
            h5_obj = h5py.File(path, 'r')
            data_shape = util.get_data_shape(path)
            # check default dataset
            if dataset not in data_shape:
                dataset = self.select_dataset(path)
                if dataset is None:
                    return
            if len(data_shape[dataset]) == 3:
                nb_frame = data_shape[dataset][0]
                shape = data_shape[dataset][1:]
            else:
                nb_frame = 1
                shape = data_shape[dataset]
            self.path = path
            self.dataset = dataset
            self.total_frames = nb_frame
            self.h5_obj = h5_obj
            self.shape = shape
        elif ext == 'lcls':  # self-defined format
            dataset = 'lcls-data'
            lcls_data = util.get_lcls_data(path)
            data_shape = util.get_data_shape(path)
            nb_frame = data_shape[dataset][0]
            self.path = path
            self.dataset = dataset
            self.total_frames = nb_frame
            self.lcls_data = lcls_data
            self.shape = data_shape[dataset][1:]
        elif ext == 'tif':
            dataset = 'tif-data'
            data_shape = util.get_data_shape(path)
            self.path = path
            self.dataset = dataset
            self.total_frames = 1
            self.shape = data_shape['tif-data']
        else:
            return

    def select_dataset(self, path):
        combo_box = self.dataset_diag.comboBox
        combo_box.clear()
        data_shape = util.get_data_shape(path)
        for dataset, shape in data_shape.items():
            combo_box.addItem('%s %s' % (dataset, shape), userData=dataset)
        if self.dataset_diag.exec_() == QDialog.Accepted:
            selected_id = combo_box.currentIndex()
            dataset = combo_box.itemData(selected_id)
            return dataset
        else:
            return None

    def maybe_add_file(self, path):
        ext = path.split('.')[-1]
        if ext not in self.accepted_file_types:
            self.add_info('Unsupported file type: %s' % path)
            return
        if not os.path.exists(path):
            self.add_info('File not exist %s' % path)
            return
        if path in self.all_files:
            self.add_info('Skip existing file %s' % path)
        else:
            self.add_info('Add %s' % path)
            basename = os.path.basename(path)
            item = QListWidgetItem()
            item.setText(basename)
            item.setData(1, path)
            item.setToolTip(path)
            self.fileList.addItem(item)
            self.all_files.append(path)

# info panel
    def add_info(self, info, info_type='INFO'):
        now = datetime.now()
        self.infoPanel.appendPlainText(
            '%s: [%s] %s' % (now.strftime('%Y-%m-%d %H:%M:%S'), info_type, info)
        )

# drag and drop
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            drop_file = url.toLocalFile()
            ext = drop_file.split('.')[-1]
            if ext in self.accepted_file_types:
                event.accept()
                return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            drop_file = url.toLocalFile()
            self.maybe_add_file(drop_file)

    def closeEvent(self, _):
        pass


def create_project(project_name, facility):
    print('Project %s for %s is created.' % (project_name, facility))
    os.makedirs((os.path.join(project_name, '.click')))
    os.makedirs(os.path.join(project_name, 'mean'))
    mask_dir = os.path.join(SOURCE_DIR, 'data', 'mask', facility)
    if os.path.exists(mask_dir):
        copytree(mask_dir, os.path.join(project_name, 'mask'))
    geom_dir = os.path.join(SOURCE_DIR, 'data', 'geom', facility)
    if os.path.exists(geom_dir):
        copytree(geom_dir, os.path.join(project_name, 'geom'))
    os.makedirs(os.path.join(project_name, 'powder'))
    os.makedirs(os.path.join(project_name, 'raw_lst'))
    os.makedirs(os.path.join(project_name, 'cxi_comp'))
    os.makedirs(os.path.join(project_name, 'cxi_hit'))
    os.makedirs(os.path.join(project_name, 'indexing'))
    os.makedirs(os.path.join(project_name, 'conf', 'hit_finding'))
    os.makedirs(os.path.join(project_name, 'conf', 'indexing'))
    copyfile(os.path.join(SOURCE_DIR, 'conf', 'config-%s.yml' % facility),
             os.path.join(project_name, '.click', 'config.yml'))
    if facility == 'LCLS':
        os.makedirs(os.path.join(project_name, 'xtc_proxy'))


if __name__ == '__main__':
    argv = docopt(__doc__)
    if argv['startproject']:
        create_project(argv['<project>'], argv['--facility'])
    elif argv['--version']:
        print('Click 1.0 by Xuanxuan Li(lxx2011011580@gmail.com).')
    else:
        # check environment
        if not os.path.exists('.click'):
            print('This is not a Click project directory!')
            sys.exit()
        app = QApplication(sys.argv)
        win = GUI()
        win.setWindowTitle('Click')
        win.showMaximized()
        sys.exit(app.exec_())
