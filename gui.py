import sys
import os
from glob import glob
from functools import partial
import yaml
from datetime import datetime

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from pyqtgraph import mkPen
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QPoint, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, \
    QMenu, QListWidgetItem, QTableWidgetItem, QWidget
from PyQt5.uic import loadUi

from skimage.morphology import disk, binary_dilation, binary_erosion
from scipy.ndimage.filters import gaussian_filter
import h5py
import numpy as np

from threads import MeanCalculatorThread, GenPowderThread
from util import util
from settings import Settings
from job_win import JobWindow
from powder_win import PowderWindow


class GUI(QMainWindow):
    def __init__(self, settings):
        super(GUI, self).__init__()
        # min gui
        self.settings = settings
        self.workdir = settings.workdir
        self.peak_size = settings.peak_size
        self.dataset_def = settings.dataset_def
        self.max_info = settings.max_info

        # setup layout
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/gui.ui' % dir_, self)
        self.info_panel.setMaximumBlockCount(self.max_info)
        self.inspector = QDialog()
        self.inspector.setWindowFlags(
            self.inspector.windowFlags() | Qt.WindowStaysOnTopHint)
        loadUi('%s/ui/inspector.ui' % dir_, self.inspector)
        self.dataset_diag = QDialog()
        loadUi('%s/ui/dataset_diag.ui' % dir_, self.dataset_diag)
        self.mean_diag = QDialog()
        loadUi('%s/ui/mean_diag.ui' % dir_, self.mean_diag)
        self.powder_diag = QDialog()
        loadUi('%s/ui/powder_diag.ui' % dir_, self.powder_diag)
        self.peak_table = QWidget()
        loadUi('%s/ui/peak_table.ui' % dir_, self.peak_table)

        self.job_win = JobWindow(settings=self.settings)
        self.powder_win = PowderWindow(settings=self.settings)

        self.gradient_view.hide()
        self.calib_mask_view.hide()
        self.splitter_2.setSizes([0.2 * self.height(), 0.8 * self.height()])
        self.splitter_3.setSizes([0.3 * self.height(), 0.7 * self.height()])
        self.splitter_4.setSizes(
            [0.25 * self.width(), 0.5 * self.width(), 0.25 * self.width()]
        )
        self.setAcceptDrops(True)

        self.show_file_list = True
        self.accepted_file_types = ('h5', 'npy', 'cxi', 'npz')
        self.curr_files = []
        self.mask_file = None
        self.file = None
        self.h5_obj = None
        self.dataset = None
        self.nb_frame = 0
        self.frame = 0

        # hit finder parameters
        self.show_view2 = False  # gradient view
        self.mask_on = True
        self.hit_finding_on = False
        self.show_raw_peaks = False
        self.show_valid_peaks = False
        self.show_opt_peaks = False
        self.show_strong_peaks = True
        self.gaussian_sigma = 1
        self.min_peak_num = 0
        self.max_peak_num = 500
        self.min_intensity = 0.
        self.min_gradient = 0.
        self.min_distance = 10
        self.peak_refine_mode_list = ['gradient', 'mean']
        self.peak_refine_mode = self.peak_refine_mode_list[0]
        self.min_snr = 0.  # min srn for a strong peak
        self.min_pixels = 2  # min pixel num for a strong peak
        self.snr_mode_list = ['rings', 'simple', 'adaptive']
        self.snr_mode = self.snr_mode_list[0]
        self.signal_radius = 1
        self.bg_inner_radius = 2
        self.bg_outer_radius = 3
        self.crop_size = 7
        self.bg_ratio = 0.7
        self.signal_ratio = 0.2
        self.signal_thres = 5.0

        # calib/mask parameters
        self.show_view3 = False  # show calib/mask view
        self.show_center = True
        self.center = np.array([0., 0.])
        self.calib_mask_threshold = 0
        self.ring_radii = np.array([])
        self.erosion1_size = 0
        self.dilation_size = 0
        self.erosion2_size = 0

        self.img = None  # raw image
        self.img2 = None  # gradient image
        self.img3 = None  # calib/mask image
        self.strong_peaks = None
        self.peak_info = None
        self.mask = None
        self.peak_item = pg.ScatterPlotItem(
            symbol='x', size=10, pen='r', brush=(255, 255, 255, 0)
        )
        self.opt_peak_item = pg.ScatterPlotItem(
            symbol='+', size=10, pen='y', brush=(255, 255, 255, 0)
        )
        self.strong_peak_item = pg.ScatterPlotItem(
            symbol='o', size=10, pen='g', brush=(255, 255, 255, 0)
        )
        self.signal_pixel_item = pg.ScatterPlotItem(
            symbol='t', size=10, pen='g', brush=(255, 255, 255, 0)
        )
        self.background_pixel_item = pg.ScatterPlotItem(
            symbol='t1', size=10, pen='r', brush=(255, 255, 255, 0)
        )
        self.center_item = pg.ScatterPlotItem()
        self.ring_item = pg.ScatterPlotItem()

        # threads
        self.calc_mean_thread = None
        self.gen_powder_thread = None

        # add plot item to image view
        self.raw_view.getView().addItem(self.peak_item)
        self.raw_view.getView().addItem(self.opt_peak_item)
        self.raw_view.getView().addItem(self.strong_peak_item)
        self.raw_view.getView().addItem(self.signal_pixel_item)
        self.raw_view.getView().addItem(self.background_pixel_item)
        self.calib_mask_view.getView().addItem(self.center_item)
        self.calib_mask_view.getView().addItem(self.ring_item)

        # status tree
        status_params = [
            {
                'name': 'filepath', 'type': 'str', 'readonly': True
            },
            {
                'name': 'dataset', 'type': 'str', 'readonly': True
            },
            {
                'name': 'mask file', 'type': 'str', 'readonly': True
            },
            {
                'name': 'total frame', 'type': 'str', 'readonly': True,
            },
            {
                'name': 'current frame', 'type': 'int', 'value': self.frame,
            }
        ]
        self.status_params = Parameter.create(
            name='status parameters',
            type='group',
            children=status_params,
        )
        self.status_tree.setParameters(
            self.status_params, showTop=False
        )

        # hit finder parameter tree
        hit_finder_params = [
            {
                'name': 'hit finding on', 'type': 'bool',
                'value': self.hit_finding_on
            },
            {
                'name': 'mask on', 'type': 'bool', 'value': self.mask_on,
                'visible': False
            },
            {
                'name': 'show raw peaks', 'type': 'bool',
                'value': self.show_raw_peaks
            },
            {
                'name': 'show valid peaks', 'type': 'bool',
                'value': self.show_valid_peaks
            },
            {
                'name': 'show opt peaks', 'type': 'bool',
                'value': self.show_opt_peaks
            },
            {
                'name': 'show strong peaks', 'type': 'bool',
                'value': self.show_strong_peaks
            },
            {
                'name': 'gaussian filter sigma', 'type': 'float',
                'value': self.gaussian_sigma
            },
            {
                'name': 'min peak num', 'type': 'int',
                'value': self.min_peak_num
            },
            {
                'name': 'max peak num', 'type': 'int',
                'value': self.max_peak_num
            },
            {
                'name': 'min gradient', 'type': 'float',
                'value': self.min_gradient
            },
            {
                'name': 'min distance', 'type': 'int',
                'value': self.min_distance
            },
            {
                'name': 'peak refine mode', 'type': 'list',
                'values': self.peak_refine_mode_list,
            },
            {
                'name': 'min snr', 'type': 'float', 'value': self.min_snr
            },
            {
                'name': 'min pixels', 'type': 'int', 'value': self.min_pixels
            },
            {
                'name': 'snr mode', 'type': 'list',
                'values': self.snr_mode_list,
            },
            {
                'name': 'signal radius', 'type': 'int',
                'value': self.signal_radius,
                'visible': False,
            },
            {
                'name': 'background inner radius', 'type': 'int',
                'value': self.bg_inner_radius,
                'visible': False,
            },
            {
                'name': 'background outer radius', 'type': 'int',
                'value': self.bg_outer_radius,
                'visible': False,
            },
            {
                'name': 'crop size', 'type': 'int',
                'value': self.crop_size,
                'visible': False,
            },
            {
                'name': 'background ratio', 'type': 'float',
                'value': self.bg_ratio,
            },
            {
                'name': 'signal ratio', 'type': 'float',
                'value': self.signal_ratio,
            },
            {
                'name': 'signal threshold', 'type': 'float',
                'value': self.signal_thres,
            }
        ]
        self.hit_finder_params = Parameter.create(
            name='hit finder parameters',
            type='group',
            children=hit_finder_params,
        )
        self.hit_finder_tree.setParameters(
            self.hit_finder_params, showTop=False
        )

        # calib parameter tree
        calib_mask_params = [
            {
                'name': 'show center', 'type': 'bool',
                'value': self.show_center,
            },
            {
                'name': 'threshold', 'type': 'float',
                'value': self.calib_mask_threshold,
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
            {
                'name': 'save mask', 'type': 'action'
            }
        ]
        self.calib_mask_params = Parameter.create(
            name='calib/mask parameters',
            type='group',
            children=calib_mask_params,
        )
        self.calib_mask_tree.setParameters(
            self.calib_mask_params, showTop=False
        )

        # menu bar action
        self.action_open.triggered.connect(self.open_file)
        self.action_save_hit_finding_conf.triggered.connect(self.save_conf)
        self.action_load_hit_finding_conf.triggered.connect(self.load_conf)
        self.action_show_inspector.triggered.connect(
            self.show_inspector
        )
        self.action_show_gradient_view.triggered.connect(
            self.show_or_hide_gradient_view
        )
        self.action_test.triggered.connect(
            self.show_or_hide_file_list
        )
        self.action_show_calib_mask_view.triggered.connect(
            self.show_or_hide_calib_mask_view
        )
        self.action_show_peak_table.triggered.connect(
            self.show_peak_table
        )
        self.action_job_table.triggered.connect(
            self.show_job_win
        )
        self.action_powder_fit.triggered.connect(
            self.show_powder_win
        )

        # peak table
        self.peak_table.peak_table.cellDoubleClicked.connect(
            self.zoom_in_on_peak
        )

        # mean/std dialog
        self.mean_diag.browse_btn.clicked.connect(
            partial(self.choose_dir, self.mean_diag.line_edit_1))
        self.mean_diag.combo_box.currentIndexChanged.connect(
            self.update_mean_diag_nframe)
        self.mean_diag.apply_btn.clicked.connect(self.calc_mean_std)

        # powder dialog
        self.powder_diag.browse_btn.clicked.connect(
            partial(self.choose_dir, self.powder_diag.line_edit_1))
        self.powder_diag.combo_box.currentIndexChanged.connect(
            self.update_powder_diag_nframe)
        self.powder_diag.submit_btn.clicked.connect(self.gen_powder)

        # status
        self.status_params.param(
            'current frame'
        ).sigValueChanged.connect(self.change_frame)

        # calib/mask
        self.calib_mask_params.param(
            'show center'
        ).sigValueChanged.connect(self.change_show_center)
        self.calib_mask_params.param(
            'threshold'
        ).sigValueChanged.connect(self.change_calib_mask_threshold)
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
            'erosion1 size'
        ).sigValueChanged.connect(self.change_erosion1_size)
        self.calib_mask_params.param(
            'dilation size'
        ).sigValueChanged.connect(self.change_dilation_size)
        self.calib_mask_params.param(
            'erosion2 size'
        ).sigValueChanged.connect(self.change_erosion2_size)
        self.calib_mask_params.param(
            'save mask'
        ).sigActivated.connect(self.save_mask)

        # file lists, image views
        self.raw_view.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=1))
        self.gradient_view.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=2))
        self.calib_mask_view.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=3))
        self.file_list.itemDoubleClicked.connect(self.load_file)
        self.file_list.customContextMenuRequested.connect(
            self.show_menu)
        self.line_edit.returnPressed.connect(self.add_file)

        # hit finder
        self.hit_finder_params.param(
            'hit finding on').sigValueChanged.connect(
            self.change_hit_finding)
        self.hit_finder_params.param(
            'mask on').sigValueChanged.connect(
            self.apply_mask)
        self.hit_finder_params.param(
            'show raw peaks').sigValueChanged.connect(
            self.change_show_raw_peaks)
        self.hit_finder_params.param(
            'show valid peaks').sigValueChanged.connect(
            self.change_show_valid_peaks)
        self.hit_finder_params.param(
            'show opt peaks').sigValueChanged.connect(
            self.change_show_opt_peaks)
        self.hit_finder_params.param(
            'show strong peaks').sigValueChanged.connect(
            self.change_show_strong_peaks)
        self.hit_finder_params.param(
            'gaussian filter sigma').sigValueChanged.connect(
            self.change_gaussian_sigma)
        self.hit_finder_params.param(
            'min peak num').sigValueChanged.connect(self.change_min_peak_num)
        self.hit_finder_params.param(
            'max peak num').sigValueChanged.connect(self.change_max_peak_num)
        self.hit_finder_params.param(
            'min gradient').sigValueChanged.connect(self.change_min_gradient)
        self.hit_finder_params.param(
            'min distance').sigValueChanged.connect(self.change_min_distance)
        self.hit_finder_params.param(
            'peak refine mode').sigValueChanged.connect(
            self.change_peak_refine_mode)
        self.hit_finder_params.param(
            'min snr').sigValueChanged.connect(self.change_min_snr)
        self.hit_finder_params.param(
            'min pixels').sigValueChanged.connect(self.change_min_pixels)
        self.hit_finder_params.param(
            'snr mode').sigValueChanged.connect(self.change_snr_mode)
        self.hit_finder_params.param(
            'signal radius').sigValueChanged.connect(self.change_signal_radius)
        self.hit_finder_params.param(
            'background inner radius').sigValueChanged.connect(
            self.change_bg_inner_radius)
        self.hit_finder_params.param(
            'background outer radius').sigValueChanged.connect(
            self.change_bg_outer_radius)
        self.hit_finder_params.param(
            'crop size').sigValueChanged.connect(self.change_crop_size)
        self.hit_finder_params.param(
            'background ratio').sigValueChanged.connect(self.change_bg_ratio)
        self.hit_finder_params.param(
            'signal ratio').sigValueChanged.connect(self.change_signal_ratio)
        self.hit_finder_params.param(
            'signal threshold').sigValueChanged.connect(
            self.change_signal_thres)

# menu slots
    @pyqtSlot()
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data (*.h5 *.cxi *.npy)")
        if len(filepath) == 0:
            return
        self.maybe_add_file(filepath)

    @pyqtSlot()
    def save_conf(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Hit Finding Conf File",
            self.workdir, "Yaml Files (*.yml)"
        )
        if len(filepath) == 0:
            return
        conf_dict = {
            'dataset': self.dataset,
            'mask file': self.mask_file,
            'gaussian filter sigma': self.gaussian_sigma,
            'min peak num': self.min_peak_num,
            'max peak num': self.max_peak_num,
            'min gradient': self.min_gradient,
            'peak refine mode': self.peak_refine_mode,
            'min distance': self.min_distance,
            'min snr': self.min_snr,
            'min pixels': self.min_pixels,
            'snr mode': self.snr_mode,
            'signal radius': self.signal_radius,
            'background inner radius': self.bg_inner_radius,
            'background outer radius': self.bg_outer_radius,
            'crop size': self.crop_size,
            'background ratio': self.bg_ratio,
            'signal ratio': self.signal_ratio,
            'signal threshold': self.signal_thres,
        }
        with open(filepath, 'w') as f:
            yaml.dump(conf_dict, f, default_flow_style=False)

    @pyqtSlot()
    def load_conf(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Hit Finding Conf File",
            self.workdir,
            "Yaml Files (*.yml)"
        )
        if len(filepath) == 0:
            return
        with open(filepath, 'r') as f:
            conf_dict = yaml.load(f)
        if 'dataset' in conf_dict.keys():
            self.dataset_def = self.dataset = conf_dict['dataset']
        if 'mask file' in conf_dict.keys():
            self.mask_file = conf_dict['mask file']
            self.status_params.param('mask file').setValue(self.mask_file)
            self.mask = util.read_image(self.mask_file)
        if 'gaussian filter sigma' in conf_dict.keys():
            self.gaussian_sigma = conf_dict['gaussian filter sigma']
            self.hit_finder_params.param(
                'gaussian filter sigma'
            ).setValue(self.gaussian_sigma)
        if 'min peak num' in conf_dict.keys():
            self.min_peak_num = conf_dict['min peak num']
            self.hit_finder_params.param(
                'min peak num'
            ).setValue(self.min_peak_num)
        if 'max peak num' in conf_dict.keys():
            self.max_peak_num = conf_dict['max peak num']
            self.hit_finder_params.param(
                'max peak num'
            ).setValue(self.max_peak_num)
        if 'min gradient' in conf_dict.keys():
            self.min_gradient = conf_dict['min gradient']
            self.hit_finder_params.param(
                'min gradient'
            ).setValue(self.min_gradient)
        if 'peak refine mode' in conf_dict.keys():
            self.peak_refine_mode = conf_dict['peak refine mode']
            self.hit_finder_params.param(
                'peak refine mode'
            ).setValue(self.peak_refine_mode)
        if 'min distance' in conf_dict.keys():
            self.min_distance = conf_dict['min distance']
            self.hit_finder_params.param(
                'min distance'
            ).setValue(self.min_distance)
        if 'min snr' in conf_dict.keys():
            self.min_snr = conf_dict['min snr']
            self.hit_finder_params.param(
                'min snr'
            ).setValue(self.min_snr)
        if 'min pixels' in conf_dict.keys():
            self.min_pixels = conf_dict['min pixels']
            self.hit_finder_params.param(
                'min pixels'
            ).setValue(self.min_pixels)
        if 'snr mode' in conf_dict.keys():
            self.snr_mode = conf_dict['snr mode']
            self.hit_finder_params.param(
                'snr mode'
            ).setValue(self.snr_mode)
        if 'signal radius' in conf_dict.keys():
            self.signal_radius = conf_dict['signal radius']
            self.hit_finder_params.param(
                'signal radius'
            ).setValue(self.signal_radius)
        if 'background inner radius' in conf_dict.keys():
            self.bg_inner_radius = conf_dict['background inner radius']
            self.hit_finder_params.param(
                'background inner radius'
            ).setValue(self.bg_inner_radius)
        if 'background outer radius' in conf_dict.keys():
            self.bg_outer_radius = conf_dict['background outer radius']
            self.hit_finder_params.param(
                'background outer radius'
            ).setValue(self.bg_outer_radius)
        if 'crop size' in conf_dict.keys():
            self.crop_size = conf_dict['crop size']
            self.hit_finder_params.param(
                'crop size'
            ).setValue(self.crop_size)
        if 'background ratio' in conf_dict.keys():
            self.bg_ratio = conf_dict['background ratio']
            self.hit_finder_params.param(
                'background ratio'
            ).setValue(self.bg_ratio)
        if 'signal ratio' in conf_dict.keys():
            self.signal_ratio = conf_dict['signal ratio']
            self.hit_finder_params.param(
                'signal ratio'
            ).setValue(self.signal_ratio)
        if 'signal threshold' in conf_dict.keys():
            self.signal_thres = conf_dict['signal threshold']
            self.hit_finder_params.param(
                'signal threshold'
            ).setValue(self.signal_thres)

    @pyqtSlot()
    def show_or_hide_gradient_view(self):
        self.show_view2 = not self.show_view2
        if self.show_view2:
            self.action_show_gradient_view.setText('Hide gradient view')
            self.gradient_view.show()
        else:
            self.action_show_gradient_view.setText('Show gradient view')
            self.gradient_view.hide()
        self.update_display()

    @pyqtSlot()
    def show_or_hide_calib_mask_view(self):
        self.show_view3 = not self.show_view3
        if self.show_view3:
            self.action_show_calib_mask_view.setText('Hide calib/mask view')
            self.calib_mask_view.show()
        else:
            self.action_show_calib_mask_view.setText('Show calib/mask view')
            self.calib_mask_view.hide()
        self.update_display()

    @pyqtSlot()
    def show_peak_table(self):
        self.peak_table.show()
        self.update_display()

    @pyqtSlot()
    def show_inspector(self):
        self.inspector.show()

    @pyqtSlot()
    def show_or_hide_file_list(self):
        self.show_file_list = not self.show_file_list
        if self.show_file_list:
            self.action_test.setText('Hide file list')
            self.file_list_frame.show()
        else:
            self.action_test.setText('Show file list')
            self.file_list_frame.hide()

    @pyqtSlot()
    def show_job_win(self):
        self.job_win.showMaximized()
        job_table = self.job_win.job_table
        width = job_table.width()
        col_count = job_table.columnCount()
        header = job_table.horizontalHeader()
        for i in range(col_count):
            header.resizeSection(i, width//col_count)

    @pyqtSlot()
    def show_powder_win(self):
        self.powder_win.show()

# peak table slots
    @pyqtSlot(int, int)
    def zoom_in_on_peak(self, row, _):
        table = self.peak_table.peak_table
        peak_id = int(table.item(row, 0).text())
        x, y = self.strong_peaks[peak_id]
        self.raw_view.getView().setRange(
            xRange=(x-20, x+20), yRange=(y-20, y+20)
        )
        if self.peak_info is not None:
            bg = self.peak_info['background values'][peak_id]
            noise = self.peak_info['noise values'][peak_id]
            self.raw_view.setLevels(bg, bg + self.signal_thres * noise)

# mean/std dialog slots
    @pyqtSlot()
    def calc_mean_std(self):
        selected_items = self.file_list.selectedItems()
        files = []
        for item in selected_items:
            files.append(item.data(1))
        dataset = self.mean_diag.combo_box.currentText()
        nb_frame = int(self.mean_diag.label_1.text())
        max_frame = min(int(self.mean_diag.spin_box.text()), nb_frame)
        output_dir = self.mean_diag.line_edit_1.text()
        prefix = self.mean_diag.line_edit_2.text()
        output = os.path.join(output_dir, '%s.npz' % prefix)

        self.calc_mean_thread = MeanCalculatorThread(
            files=files, dataset=dataset, max_frame=max_frame, output=output
        )
        self.calc_mean_thread.update_progress.connect(
            self.update_progressbar
        )
        self.calc_mean_thread.finished.connect(
            self.calc_mean_finished
        )
        self.calc_mean_thread.start()

    @pyqtSlot()
    def choose_dir(self, line_edit=None):
        if line_edit is not None:
            curr_dir = line_edit.text()
        else:
            curr_dir = ""
        dir_ = QFileDialog.getExistingDirectory(
            self, "Choose directory", curr_dir
        )
        if line_edit is not None:
            line_edit.setText(dir_)

    @pyqtSlot(float)
    def update_progressbar(self, val):
        self.mean_diag.progress_bar.setValue(val)

    @pyqtSlot()
    def calc_mean_finished(self):
        self.add_info('Mean/sigma calculation done.')

    @pyqtSlot(int)
    def update_mean_diag_nframe(self, curr_index):
        if curr_index == -1:
            return
        dataset = self.mean_diag.combo_box.itemText(curr_index)
        selected_items = self.file_list.selectedItems()
        nb_frame = 0
        for item in selected_items:
            filepath = item.data(1)
            try:
                data_shape = util.get_data_shape(filepath)
                nb_frame += data_shape[dataset][0]
            except IOError:
                self.add_info('Failed to open %s' % filepath)
                pass
        self.mean_diag.label_1.setText(str(nb_frame))

# peak powder dialog slots
    @pyqtSlot(int)
    def update_powder_diag_nframe(self, _):
        tag = self.powder_diag.combo_box.currentText()
        conf_file = os.path.join(self.workdir, 'conf/%s.yml' % tag)
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)
        dataset = conf['dataset']
        selected_items = self.file_list.selectedItems()
        nb_frame = 0
        for item in selected_items:
            filepath = item.data(1)
            try:
                data_shape = util.get_data_shape(filepath)
                nb_frame += data_shape[dataset][0]
            except IOError:
                self.add_info('Failed to open %s' % filepath)
                pass
        self.powder_diag.label_1.setText(str(nb_frame))

    @pyqtSlot()
    def gen_powder(self):
        print('generate peak powder')
        selected_items = self.file_list.selectedItems()
        files = []
        for item in selected_items:
            files.append(item.data(1))
        powder_diag = self.powder_diag
        tag = powder_diag.combo_box.currentText()
        conf_file = os.path.join(self.workdir, 'conf/%s.yml' % tag)
        nb_frame = int(powder_diag.label_1.text())
        max_frame = min(int(powder_diag.spin_box.text()), nb_frame)
        output_dir = powder_diag.line_edit_1.text()
        prefix = powder_diag.line_edit_2.text()
        output = os.path.join(output_dir, '%s.npz' % prefix)

        self.gen_powder_thread = GenPowderThread(
            files, conf_file, self.settings,
            max_frame=max_frame,
            output=output,
        )
        self.gen_powder_thread.info.connect(self.add_info)
        self.gen_powder_thread.start()

# file list / image view slots
    @pyqtSlot(QPoint)
    def show_menu(self, pos):
        menu = QMenu()
        item = self.file_list.currentItem()
        if not isinstance(item, QListWidgetItem):
            return
        filepath = item.data(1)
        ext = filepath.split('.')[-1]
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
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        if action == action_select_and_load_dataset:
            if ext == 'npy':
                self.add_info(
                    'Unsupported file type for dataset selection: %s' % ext
                )
                return  # ignore npy file for dataset selection
            data_shape = util.get_data_shape(filepath)
            dataset = self.select_dataset(filepath)
            self.nb_frame = data_shape[dataset][0]
            if filepath.split('.')[-1] in ('cxi', 'h5'):
                self.h5_obj = h5py.File(filepath, 'r')
            self.file = filepath
            self.dataset = dataset
            # update file info and display
            self.status_params.param('filepath').setValue(filepath)
            self.status_params.param('dataset').setValue(self.dataset)
            self.status_params.param('total frame').setValue(self.nb_frame)
            self.change_image()
        elif action == action_set_as_mask:
            self.mask_file = filepath
            self.mask = util.read_image(filepath)
            self.status_params.param('mask file').setValue(filepath)
        elif action == action_multiply_masks:
            items = self.file_list.selectedItems()
            mask_files = []
            for item in items:
                filepath = item.data(1)
                if filepath.split('.')[-1] == 'npy':
                    mask_files.append(filepath)
            if len(mask_files) == 0:
                return
            save_file, _ = QFileDialog.getSaveFileName(
                self, "Save mask", self.workdir, "npy file(*.npy)"
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
            combo_box = self.mean_diag.combo_box
            combo_box.clear()
            data_shape = util.get_data_shape(filepath)
            for dataset, shape in data_shape.items():
                combo_box.addItem(dataset)
            output_dir = os.path.join(self.workdir, 'mean')
            self.mean_diag.line_edit_1.setText(output_dir)
            self.mean_diag.progress_bar.setValue(0)
            self.mean_diag.exec_()

        elif action == action_gen_powder:
            conf_dir = os.path.join(self.workdir, 'conf')
            confs = glob('%s/*.yml' % conf_dir)
            self.powder_diag.combo_box.clear()
            for conf in confs:
                tag = os.path.basename(conf).split('.')[0]
                self.powder_diag.combo_box.addItem(tag)
            output_dir = os.path.join(self.workdir, 'powder')
            self.powder_diag.line_edit_1.setText(output_dir)
            self.powder_diag.exec_()
        elif action == action_del_file:
            items = self.file_list.selectedItems()
            for item in items:
                row = self.file_list.row(item)
                self.file_list.takeItem(row)
                self.curr_files.remove(item.data(1))
                self.add_info('Remove %s' % item.data(1))

        self.update_display()

    @pyqtSlot()
    def add_file(self):
        files = glob(self.line_edit.text(), recursive=True)
        for f in files:
            self.maybe_add_file(f)

    @pyqtSlot('QListWidgetItem*')
    def load_file(self, file_item):
        self.add_info('Loading %s' % file_item.data(1))
        filepath = file_item.data(1)
        ext = QtCore.QFileInfo(filepath).suffix()
        if ext == 'npy':
            self.file = filepath
            self.nb_frame = 1
        elif ext == 'npz':
            data_shape = util.get_data_shape(filepath)
            if self.dataset_def not in data_shape.keys():
                dataset = self.select_dataset(filepath)
                if len(dataset) == 0:
                    return
                self.file = filepath
                self.dataset = dataset
                if len(data_shape[dataset]) == 3:
                    self.nb_frame = data_shape[dataset][0]
                else:
                    self.nb_frame = 1
        elif ext in ('h5', 'cxi'):
            h5_obj = h5py.File(filepath, 'r')
            data_shape = util.get_data_shape(filepath)
            # check default dataset
            if self.dataset_def not in data_shape.keys():
                dataset = self.select_dataset(filepath)
                if len(dataset) == 0:
                    return
            else:
                dataset = self.dataset_def
            # PAL specific h5 file
            if 'header/frame_num' in h5_obj.keys():
                self.file = filepath
                self.h5_obj = h5_obj
                self.dataset = dataset
                if len(data_shape[dataset]) == 3:
                    self.nb_frame = data_shape[dataset][0]
                else:
                    self.nb_frame = 1
            if dataset in h5_obj:
                self.file = filepath
                self.h5_obj = h5_obj
                self.dataset = dataset
                if len(data_shape[dataset]) == 3:
                    self.nb_frame = data_shape[dataset][0]
                else:
                    self.nb_frame = 1
        else:
            return
        # update file info and display
        self.status_params.param('filepath').setValue(filepath)
        self.status_params.param('dataset').setValue(self.dataset)
        self.status_params.param('total frame').setValue(self.nb_frame)
        self.change_image()
        self.update_display()

    @pyqtSlot(object)
    def mouse_moved(self, pos, flag=None):
        if self.file is None:
            return
        if flag == 1:  # in raw image view
            mouse_point = self.raw_view.view.mapToView(pos)
        elif flag == 2:  # in gradient image view
            mouse_point = self.gradient_view.view.mapToView(pos)
        elif flag == 3:  # in calib/mask view
            mouse_point = self.calib_mask_view.view.mapToView(pos)
        else:
            return
        x, y = int(mouse_point.x()), int(mouse_point.y())
        if 0 <= x < self.img.shape[0] and 0 <= y < self.img.shape[1]:
            message = 'x:%d y:%d, I(raw): %.2E;' % (x, y, self.img[x, y])
            if self.show_view2 and self.img2 is not None:
                message += 'I(gradient): %.2E' % self.img2[x, y]
            if self.show_view3 and self.img3 is not None:
                message += 'I(calib/mask): %.2E' % self.img3[x, y]
            self.statusbar.showMessage(message, 5000)
        else:
            return
        if self.inspector.isVisible():  # show data inspector
            # out of bound check
            if x - 3 < 0 or x + 4 > self.img.shape[0]:
                return
            elif y - 3 < 0 or y + 4 > self.img.shape[1]:
                return
            # calculate snr
            pos = np.reshape((x, y), (-1, 2))
            snr_info = util.calc_snr(
                self.img, pos,
                mode=self.snr_mode,
                signal_radius=self.signal_radius,
                bg_inner_radius=self.bg_inner_radius,
                bg_outer_radius=self.bg_outer_radius,
                crop_size=self.crop_size,
                bg_ratio=self.bg_ratio,
                signal_ratio=self.signal_ratio,
                signal_thres=self.signal_thres,
                label_pixels=True,
            )
            self.inspector.snr_label.setText('SNR@(%d, %d):' % (x, y))
            self.inspector.snr_value.setText(
                '%.1f(sig %.1f, bg %.1f, noise %.1f)' %
                (snr_info['snr'][0],
                 snr_info['signal values'][0],
                 snr_info['background values'][0],
                 snr_info['noise values'][0]))
            # set table values
            signal_pixels = (snr_info['signal pixels'] - pos + 3).tolist()
            BG_pixels = (snr_info['background pixels'] - pos + 3).tolist()
            for i in range(7):
                for j in range(7):
                    v1 = self.img[x + i - 3, y + j - 3]
                    if self.show_view2:
                        v2 = self.img2[x + i - 3, y + j - 3]
                        item = QTableWidgetItem('%d\n%d' % (v1, v2))
                    else:
                        item = QTableWidgetItem('%d' % v1)
                    item.setTextAlignment(Qt.AlignCenter)
                    if [i, j] in signal_pixels:
                        item.setBackground(QtGui.QColor(178, 247, 143))
                    elif [i, j] in BG_pixels:
                        item.setBackground(QtGui.QColor(165, 173, 186))
                    else:
                        item.setBackground(QtGui.QColor(255, 255, 255))
                    self.inspector.data_table.setItem(j, i, item)

# calib/mask slots
    @pyqtSlot(object, object)
    def change_show_center(self, _, show_center):
        self.show_center = show_center
        self.update_display()

    @pyqtSlot(object, object)
    def change_calib_mask_threshold(self, _, threshold):
        self.calib_mask_threshold = threshold
        self.img3 = (self.img > threshold).astype(np.int)
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
        self.ring_radii = np.array(list(map(float, radii_str.split(','))))
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

    @pyqtSlot()
    def save_mask(self):
        if self.img3 is None:
            self.add_info('No mask image available')
        else:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save mask to", self.workdir, "npy file(*.npy)"
            )
            if len(filepath) == 0:
                return
            np.save(filepath, self.img3)
            self.add_info('Mask saved to %s' % filepath)

# hit finder slots
    @pyqtSlot(object, object)
    def apply_mask(self, _, mask_on):
        self.mask_on = mask_on
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_raw_peaks(self, _, show_raw_peaks):
        self.show_raw_peaks = show_raw_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_valid_peaks(self, _, show_valid_peaks):
        self.show_valid_peaks = show_valid_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_opt_peaks(self, _, show_opt_peaks):
        self.show_opt_peaks = show_opt_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_strong_peaks(self, _, show_strong_peaks):
        self.show_strong_peaks = show_strong_peaks
        self.update_display()

    @pyqtSlot(object, object)
    def change_hit_finding(self, _, hit_finding_on):
        self.hit_finding_on = hit_finding_on
        self.update_display()

    @pyqtSlot(object, object)
    def change_gaussian_sigma(self, _, gaussian_sigma):
        self.gaussian_sigma = gaussian_sigma
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_peak_num(self, _, min_peak_num):
        self.min_peak_num = min_peak_num
        self.update_display()

    @pyqtSlot(object, object)
    def change_max_peak_num(self, _, max_peak_num):
        self.max_peak_num = max_peak_num
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
    def change_peak_refine_mode(self, _, mode):
        self.peak_refine_mode = mode
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_snr(self, _, min_snr):
        self.min_snr = min_snr
        self.update_display()

    @pyqtSlot(object, object)
    def change_min_pixels(self, _, min_pixels):
        self.min_pixels = min_pixels
        self.update_display()

    @pyqtSlot(object, object)
    def change_snr_mode(self, _, mode):
        self.snr_mode = mode
        self.update_display()

    @pyqtSlot(object, object)
    def change_signal_radius(self, _, signal_radius):
        self.signal_radius = signal_radius
        self.update_display()

    @pyqtSlot(object, object)
    def change_bg_inner_radius(self, _, bg_inner_radius):
        self.bg_inner_radius = bg_inner_radius
        self.update_display()

    @pyqtSlot(object, object)
    def change_bg_outer_radius(self, _, bg_outer_radius):
        self.bg_outer_radius = bg_outer_radius
        self.update_display()

    @pyqtSlot(object, object)
    def change_crop_size(self, _, crop_size):
        self.crop_size = crop_size
        self.update_display()

    @pyqtSlot(object, object)
    def change_bg_ratio(self, _, bg_ratio):
        self.bg_ratio = bg_ratio
        self.update_display()

    @pyqtSlot(object, object)
    def change_signal_ratio(self, _, signal_ratio):
        self.signal_ratio = signal_ratio
        self.update_display()

    @pyqtSlot(object, object)
    def change_signal_thres(self, _, signal_thres):
        self.signal_thres = signal_thres
        self.update_display()

# status slots
    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > self.nb_frame - 1:
            frame = self.nb_frame - 1
        self.frame = frame
        self.status_params.param(
            'current frame'
        ).setValue(self.frame)
        self.change_image()
        self.update_display()

    def select_dataset(self, filepath):
        combo_box = self.dataset_diag.combo_box
        combo_box.clear()
        data_shape = util.get_data_shape(filepath)
        for dataset, shape in data_shape.items():
            combo_box.addItem('%s  %s' % (dataset, shape), userData=dataset)
        if self.dataset_diag.exec_() == QDialog.Accepted:
            id_select = combo_box.currentIndex()
            dataset = combo_box.itemData(id_select)
            if self.dataset_diag.check_box.isChecked():
                self.dataset_def = dataset
            return dataset
        else:
            return ''

    def change_image(self):
        self.img = util.read_image(
            self.file, frame=self.frame,
            h5_obj=self.h5_obj, dataset=self.dataset
        ).astype(np.float32)
        if self.gaussian_sigma > 0:
            img2 = gaussian_filter(self.img, self.gaussian_sigma)
        else:
            img2 = self.img.copy()
        grad = np.gradient(img2.astype(np.float32))
        self.img2 = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
        img3 = (self.img > self.calib_mask_threshold).astype(np.int)
        if self.erosion1_size > 0:
            selem = disk(self.erosion1_size)
            img3 = binary_erosion(img3, selem)
        if self.dilation_size > 0:
            selem = disk(self.dilation_size)
            img3 = binary_dilation(img3, selem)
        if self.erosion2_size > 0:
            selem = disk(self.erosion2_size)
            img3 = binary_erosion(img3, selem)
        self.img3 = img3

    def update_display(self):
        if self.img is None:
            return
        self.raw_view.setImage(
            self.img, autoRange=False, autoLevels=False,
            autoHistogramRange=False
        )
        self.gradient_view.setImage(
            self.img2, autoRange=False, autoLevels=False,
            autoHistogramRange=False)

        # clear all plot items
        self.center_item.clear()
        self.ring_item.clear()
        self.peak_item.clear()
        self.opt_peak_item.clear()
        self.strong_peak_item.clear()

        if self.hit_finding_on:
            peaks_dict = util.find_peaks(
                self.img, self.mask,
                gaussian_sigma=self.gaussian_sigma,
                min_gradient=self.min_gradient,
                min_distance=self.min_distance,
                max_peaks=self.max_peak_num,
                min_snr=self.min_snr,
                min_pixels=self.min_pixels,
                refine_mode=self.peak_refine_mode,
                snr_mode=self.snr_mode,
                signal_radius=self.signal_radius,
                bg_inner_radius=self.bg_inner_radius,
                bg_outer_radius=self.bg_outer_radius,
                crop_size=self.crop_size,
                bg_ratio=self.bg_ratio,
                signal_ratio=self.signal_ratio,
                signal_thres=self.signal_thres,
                label_pixels=False,
            )
            raw_peaks = peaks_dict['raw']
            if raw_peaks is not None and self.show_raw_peaks:
                self.add_info('%d raw peaks found' % len(raw_peaks))
            valid_peaks = peaks_dict['valid']
            if valid_peaks is not None and self.show_valid_peaks:
                self.add_info(
                    '%d peaks remaining after mask cleaning'
                    % len(peaks_dict['valid'])
                )
                self.peak_item.setData(pos=valid_peaks + 0.5)
            # refine peak position
            opt_peaks = peaks_dict['opt']
            if opt_peaks is not None and self.show_opt_peaks:
                self.opt_peak_item.setData(pos=opt_peaks + 0.5)
            # filtering weak peak
            self.strong_peaks = peaks_dict['strong']
            if self.strong_peaks is not None and self.show_strong_peaks:
                self.add_info('%d strong peaks' % (len(self.strong_peaks)))
                if len(self.strong_peaks) > 0:
                    self.strong_peak_item.setData(pos=self.strong_peaks + 0.5)
        if self.show_view3:
            self.calib_mask_view.setImage(
                self.img3, autoRange=False, autoLevels=False,
                autoHistogramRange=False
            )
        # show center or not
        if self.show_center:
            self.center_item.setData(
                pos=self.center.reshape(1, 2) + 0.5, symbol='+', size=24,
                pen='g', brush=(255, 255, 255, 0)
            )
        # show rings or not
        if len(self.ring_radii) > 0:
            centers = np.repeat(
                self.center.reshape(1, 2), len(self.ring_radii), axis=0)
            self.ring_item.setData(
                pos=centers + 0.5, size=self.ring_radii * 2., symbol='o',
                pen=mkPen(width=3, color='y', style=QtCore.Qt.DotLine),
                brush=(255, 255, 255, 0),
                pxMode=False,
            )
        # update peak table if visible
        if 'peaks_dict' in locals():
            self.peak_info = peaks_dict['info']
            if self.peak_table.isVisible():
                self.update_peak_table(self.peak_info)

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
        ])
        table.setData(data)

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

    def maybe_add_file(self, filepath):
        ext = filepath.split('.')[-1]
        if ext in self.accepted_file_types:
            if os.path.exists(filepath):
                if filepath in self.curr_files:
                    self.add_info('Skip existing file %s' % filepath)
                else:
                    self.add_info('Add %s' % filepath)
                    basename = os.path.basename(filepath)
                    item = QListWidgetItem()
                    item.setText(basename)
                    item.setData(1, filepath)
                    item.setToolTip(filepath)
                    self.file_list.addItem(item)
                    self.curr_files.append(filepath)
            else:
                self.add_info('File not exist %s' % filepath)
        else:
            self.add_info('Unsupported file type: %s' % filepath)

    def add_info(self, info):
        now = datetime.now()
        self.info_panel.appendPlainText(
            '[%s]: %s' % (f'{now:%Y-%m-%d %H:%M:%S}', info)
        )


def main():
    if len(sys.argv) > 1:
        print('using setting from %s' % sys.argv[1])
        with open(sys.argv[1], 'r') as f:
            settings = Settings(yaml.load(f))
    else:
        settings = Settings()
        print('using default settings')
    app = QApplication(sys.argv)
    win = GUI(settings=settings)
    win.setWindowTitle('SFX Suite')
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
