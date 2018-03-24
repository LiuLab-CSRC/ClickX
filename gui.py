import sys
from functools import partial

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from pyqtgraph import mkPen
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QDialog, QFileDialog, QMenu
from PyQt5.QtWidgets import QListWidgetItem, QTableWidgetItem
from PyQt5.uic import loadUi

from util import *
from threads import *
from job_win import JobWindow
import yaml


PEAK_SIZE = 12


class GUI(QMainWindow):
    def __init__(self, *args):
        super(GUI, self).__init__(*args)

        # setup layout
        loadUi('ui/gui.ui', self)
        self.inspector = QDialog()
        self.inspector.setWindowFlags(
            self.inspector.windowFlags() | Qt.WindowStaysOnTopHint)
        loadUi('ui/inspector.ui', self.inspector)
        self.dataset_diag = QDialog()
        loadUi('ui/dataset_diag.ui', self.dataset_diag)
        self.mean_diag = QDialog()
        loadUi('ui/mean_std.ui', self.mean_diag)
        self.job_win = JobWindow()

        self.gradient_view.hide()
        self.calib_mask_view.hide()
        self.splitter_2.setSizes([0.2 * self.height(), 0.8 * self.height()])
        self.splitter_3.setSizes([0.3 * self.height(), 0.7 * self.height()])
        self.splitter_4.setSizes(
            [0.25 * self.width(), 0.5 * self.width(), 0.25 * self.width()]
        )
        self.setAcceptDrops(True)

        self.accepted_file_types = ('h5', 'npy', 'cxi')
        self.mask_file = None
        self.file = None
        self.h5_obj = None
        self.h5_dataset = None
        self.h5_dataset_def = ''
        self.nb_frame = 0
        self.frame = 0

        # hit finder parameters
        self.show_view2 = False  # gradient view
        self.show_inspector = False
        self.mask_on = False
        self.hit_finding_on = False
        self.gaussian_sigma = 1
        self.min_snr = 0.
        self.min_peak_num = 0
        self.max_peak_num = 500
        self.min_intensity = 0.
        self.min_gradient = 0.
        self.min_distance = 10

        # calib/mask parameters
        self.show_view3 = False  # show calib/mask view
        self.show_center = True
        self.center = np.array([0., 0.])
        self.calib_mask_threshold = 0
        self.ring_radii = np.array([])

        self.img = None  # raw image
        self.img2 = None  # gradient image
        self.img3 = None  # calib/mask image
        self.mask = None
        self.peak_item = pg.ScatterPlotItem()
        self.opt_peak_item = pg.ScatterPlotItem()
        self.strong_peak_item = pg.ScatterPlotItem()
        self.center_item = pg.ScatterPlotItem()
        self.ring_item = pg.ScatterPlotItem()

        # threads
        self.calc_mean_thread = None

        # add plot item to image view
        self.raw_view.getView().addItem(self.peak_item)
        self.raw_view.getView().addItem(self.opt_peak_item)
        self.raw_view.getView().addItem(self.strong_peak_item)
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
                {'name': 'hit finding on', 'type': 'bool',
                    'value': self.hit_finding_on},
                {'name': 'mask on', 'type': 'bool',
                    'value': self.mask_on},
                {'name': 'gaussian filter sigma', 'type': 'float',
                 'value': self.gaussian_sigma},
                {'name': 'min peak num', 'type': 'int',
                    'value': self.min_peak_num},
                {'name': 'max peak num', 'type': 'int',
                    'value': self.max_peak_num},
                {'name': 'min gradient', 'type': 'float',
                 'value': self.min_gradient},
                {'name': 'min distance', 'type': 'int',
                 'value': self.min_distance},
                {'name': 'min snr', 'type': 'float',
                 'value': self.min_snr},
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
                'name': 'center x', 'type': 'float',
                'value': self.center[0]
            },
            {
                'name': 'center y', 'type': 'float',
                'value': self.center[1]
            },
            {
                'name': 'radii of rings', 'type': 'str',
                'value': ''
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
            self.show_or_hide_inspector
        )
        self.action_show_gradient_view.triggered.connect(
            self.show_or_hide_gradient_view
        )
        self.action_show_calib_mask_view.triggered.connect(
            self.show_or_hide_calib_mask_view
        )
        self.action_job_table.triggered.connect(
            self.show_job_table
        )

        # mean/std dialog
        self.mean_diag.apply_btn.clicked.connect(self.calc_mean_std)
        self.mean_diag.combo_box.currentIndexChanged.connect(
            self.update_mean_diag_nframe)

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
            'min snr').sigValueChanged.connect(self.change_min_snr)

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
            self, "Save Hit Finding Conf File", "", "Yaml Files (*.yml)")
        if len(filepath) == 0:
            return
        conf_dict = {
            'dataset': self.h5_dataset,
            'mask file': self.mask_file,
            'gaussian filter sigma': self.gaussian_sigma,
            'min peak num': self.min_peak_num,
            'max peak num': self.max_peak_num,
            'min gradient': self.min_gradient,
            'min distance': self.min_distance,
            'min snr': self.min_snr,
        }
        with open(filepath, 'w') as f:
            yaml.dump(conf_dict, f, default_flow_style=False)

    @pyqtSlot()
    def load_conf(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Hit Finding Conf File", "", "Yaml Files (*.yml)")
        if len(filepath) == 0:
            return
        with open(filepath, 'r') as f:
            conf_dict = yaml.load(f)
        if 'dataset' in conf_dict.keys():
            self.h5_dataset_def = self.h5_dataset = conf_dict['dataset']
        if 'mask file' in conf_dict.keys():
            self.mask_file = conf_dict['mask file']
            self.status_params.param('mask file').setValue(self.mask_file)
            self.mask = read_image(self.mask_file)
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
    def show_or_hide_inspector(self):
        self.show_inspector = not self.show_inspector
        if self.show_inspector:
            self.action_show_inspector.setText('Hide inspector')
            self.inspector.show()
        else:
            self.action_show_inspector.setText('Show inspector')
            self.inspector.hide()

    @pyqtSlot()
    def show_job_table(self):
        self.job_win.showMaximized()
        job_table = self.job_win.job_table
        width = job_table.width()
        col_count = job_table.columnCount()
        header = job_table.horizontalHeader()
        for i in range(col_count):
            header.resizeSection(i, width//col_count)

# mean/std diag slots
    @pyqtSlot()
    def calc_mean_std(self):
        selected_items = self.file_list.selectedItems()
        files = []
        for item in selected_items:
            files.append(item.text())
        dataset = self.mean_diag.combo_box.currentText()
        nb_frame = int(self.mean_diag.nb_frame.text())
        max_frame = min(int(self.mean_diag.max_frame.text()), nb_frame)
        prefix = self.mean_diag.prefix.text()

        self.calc_mean_thread = CalcMeanThread(
            files=files, dataset=dataset, max_frame=max_frame,
            prefix=prefix,
        )
        self.calc_mean_thread.update_progress.connect(
            self.update_progressbar
        )
        self.calc_mean_thread.start()

    @pyqtSlot(float)
    def update_progressbar(self, val):
        self.mean_diag.progress_bar.setValue(val)

    @pyqtSlot(int)
    def update_mean_diag_nframe(self, curr_index):
        if curr_index == -1:
            return
        dataset = self.mean_diag.combo_box.itemText(curr_index)
        selected_items = self.file_list.selectedItems()
        nb_frame = 0
        for item in selected_items:
            try:
                f = h5py.File(item.text(), 'r')
                shape = f[dataset].shape
                if len(shape) == 3:
                    nb_frame += shape[0]
                else:
                    nb_frame += 1
            except IOError:
                pass
        self.mean_diag.nb_frame.setText(str(nb_frame))

# file list / image view slots
    @pyqtSlot(QPoint)
    def show_menu(self, pos):
        menu = QMenu()
        item = self.file_list.currentItem()
        if not isinstance(item, QListWidgetItem):
            return
        filepath = item.text()
        action_set_as_mask = menu.addAction('set as mask')
        action_select_and_load_dataset = menu.addAction(
            'select and load dataset'
        )
        action_calc_mean_std = menu.addAction('calculate mean/std')
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        if action == action_set_as_mask:
            self.mask_file = filepath
            self.mask = read_image(filepath)
            self.status_params.param('mask file').setValue(filepath)
        elif action == action_select_and_load_dataset:
            h5_obj = h5py.File(filepath, 'r')
            h5_dataset = self.select_dataset(filepath)
            if len(h5_obj[h5_dataset].shape) == 3:
                self.nb_frame = h5_obj[h5_dataset].shape[0]
            else:
                self.nb_frame = 1
            self.file = filepath
            self.h5_obj = h5_obj
            self.h5_dataset = h5_dataset
            # update file info and display
            self.status_params.param('filepath').setValue(filepath)
            self.status_params.param('dataset').setValue(self.h5_dataset)
            self.status_params.param('total frame').setValue(self.nb_frame)
            self.change_image()
        elif action == action_calc_mean_std:
            combo_box = self.mean_diag.combo_box
            combo_box.clear()
            data_info = get_h5_info(filepath)
            for i in range(len(data_info)):
                combo_box.addItem(str(data_info[i]['key']))
            self.mean_diag.progress_bar.setValue(0)
            self.mean_diag.exec_()

        self.update_display()

    @pyqtSlot()
    def add_file(self):
        self.maybe_add_file(self.line_edit.text())

    @pyqtSlot('QListWidgetItem*')
    def load_file(self, file_item):
        self.info_panel.append('loading %s' % file_item.text())
        filepath = file_item.text()
        ext = QtCore.QFileInfo(filepath).suffix()
        if ext == 'npy':
            self.file = filepath
            self.nb_frame = 1
        elif ext in ('h5', 'cxi'):
            h5_obj = h5py.File(filepath, 'r')
            if self.h5_dataset_def not in h5_obj:  # check default dataset
                h5_dataset = self.select_dataset(filepath)
                if len(h5_dataset) == 0:
                    return
            else:
                h5_dataset = self.h5_dataset_def
            if h5_dataset in h5_obj:
                self.file = filepath
                self.h5_obj = h5_obj
                self.h5_dataset = h5_dataset
                if len(h5_obj[h5_dataset].shape) == 3:
                    self.nb_frame = h5_obj[h5_dataset].shape[0]
                else:
                    self.nb_frame = 1
        else:
            return
        # update file info and display
        self.status_params.param('filepath').setValue(filepath)
        self.status_params.param('dataset').setValue(self.h5_dataset)
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
        if self.show_inspector:  # show data inspector
            # out of bound check
            if x - 3 < 0 or x + 4 > self.img.shape[0]:
                return
            elif y - 3 < 0 or y + 4 > self.img.shape[1]:
                return
            # calculate snr
            pos = np.reshape((x, y), (-1, 2))
            snr = calc_snr(self.img, pos)
            self.inspector.snr_label.setText('SNR@(%d, %d):' % (x, y))
            self.inspector.snr_value.setText('%.1f' % snr)
            # set table values
            for i in range(5):
                for j in range(5):
                    v1 = self.img[x + i - 2, y + j - 2]
                    if self.show_view2:
                        v2 = self.img2[x + i - 2, y + j - 2]
                        item = QTableWidgetItem('%d\n%d' % (v1, v2))
                    else:
                        item = QTableWidgetItem('%d' % v1)
                    item.setTextAlignment(Qt.AlignCenter)
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

# hit finder slots
    @pyqtSlot(object, object)
    def apply_mask(self, _, mask_on):
        self.mask_on = mask_on
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
    def change_min_snr(self, _, min_snr):
        self.min_snr = min_snr
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
        data_info = get_h5_info(filepath)
        for i in range(len(data_info)):
            combo_box.addItem(
                '%s  %s' %
                (data_info[i]['key'], data_info[i]['shape']),
                userData=data_info[i]['key'])
        if self.dataset_diag.exec_() == QDialog.Accepted:
            id_select = combo_box.currentIndex()
            h5_dataset = combo_box.itemData(id_select)
            if self.dataset_diag.check_box.isChecked():
                self.h5_dataset_def = h5_dataset
            return h5_dataset
        else:
            return ''

    def change_image(self):
        self.img = read_image(
            self.file, frame=self.frame,
            h5_obj=self.h5_obj, h5_dataset=self.h5_dataset
        ).astype(np.float32)
        # smoothed gradient image
        if self.gaussian_sigma > 0:
            self.img2 = gaussian_filter(self.img, self.gaussian_sigma)
        else:
            self.img2 = self.img.copy()
        grad = np.gradient(self.img2.astype(np.float32))
        self.img2 = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
        self.img3 = (self.img > self.calib_mask_threshold).astype(np.int)

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
            peaks_dict = find_peaks(
                self.img, self.mask,
                gaussian_sigma=self.gaussian_sigma,
                min_gradient=self.min_gradient,
                min_distance=self.min_distance,
                max_peaks=self.max_peak_num,
                min_snr=self.min_snr,
            )
            raw_peaks = peaks_dict['raw']
            if raw_peaks is not None:
                self.info_panel.append('%d raw peaks found' % len(raw_peaks))
            valid_peaks = peaks_dict['valid']
            if valid_peaks is not None:
                self.info_panel.append(
                    '%d peaks remaining after mask cleaning' % len(peaks_dict['valid'])
                )
                self.peak_item.setData(
                    pos=valid_peaks + 0.5, symbol='x', size=PEAK_SIZE,
                    pen='r', brush=(255, 255, 255, 0)
                )
            # refine peak position
            opt_peaks = peaks_dict['opt']
            if opt_peaks is not None:
                self.opt_peak_item.setData(
                    pos=opt_peaks + 0.5, symbol='+', size=PEAK_SIZE,
                    pen='y', brush=(255, 255, 255, 0)
                )
            # filtering weak peak
            strong_peaks = peaks_dict['strong']
            if strong_peaks is not None:
                self.info_panel.append('%d strong peaks' % (len(strong_peaks)))
                if len(strong_peaks) > 0:
                    self.strong_peak_item.setData(
                        pos=strong_peaks + 0.5, symbol='o', size=PEAK_SIZE,
                        pen='g', brush=(255, 255, 255, 0))
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
                self.file_list.addItem(filepath)
            else:
                self.info_panel.append('File not exist %s' % filepath)
        else:
            self.info_panel.append('Unsupported file type: %s' % filepath)


def main():
    app = QApplication(sys.argv)
    win = GUI()
    win.setWindowTitle('SFX Suite')
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
