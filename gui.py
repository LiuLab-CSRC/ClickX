import sys
from functools import partial

import pyqtgraph as pg

from pyqtgraph.parametertree import Parameter
from pyqtgraph import mkPen
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QDialog, QFileDialog, QMenu
from PyQt5.QtWidgets import QListWidgetItem, QTableWidgetItem
from PyQt5.uic import loadUi

from skimage.morphology import disk, binary_dilation, binary_erosion
from threads import *
from util import *
from settings import Settings
from job_win import JobWindow
import yaml
from datetime import datetime


class GUI(QMainWindow):
    def __init__(self, parent=None, settings=None):
        super(GUI, self).__init__(parent)
        # min gui
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
        loadUi('%s/ui/mean_std.ui' % dir_, self.mean_diag)

        self.job_win = JobWindow(settings=settings)

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
        self.erosion1_size = 0
        self.dilation_size = 0
        self.erosion2_size = 0

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
                {
                    'name': 'hit finding on', 'type': 'bool',
                    'value': self.hit_finding_on
                },
                {
                    'name': 'mask on', 'type': 'bool', 'value': self.mask_on
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
                    'name': 'min snr', 'type': 'float', 'value': self.min_snr
                },
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
            self.show_or_hide_inspector
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
        self.action_job_table.triggered.connect(
            self.show_job_table
        )

        # mean/std dialog
        self.mean_diag.apply_btn.clicked.connect(self.calc_mean_std)
        self.mean_diag.browse_btn.clicked.connect(self.choose_dir)
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
            self.dataset_def = self.dataset = conf_dict['dataset']
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
    def show_or_hide_file_list(self):
        self.show_file_list = not self.show_file_list
        if self.show_file_list:
            self.action_test.setText('Hide file list')
            self.file_list_frame.show()
        else:
            self.action_test.setText('Show file list')
            self.file_list_frame.hide()

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
            files.append(item.data(1))
        dataset = self.mean_diag.combo_box.currentText()
        nb_frame = int(self.mean_diag.nb_frame.text())
        max_frame = min(int(self.mean_diag.max_frame.text()), nb_frame)
        output_dir = self.mean_diag.output_dir.text()
        prefix = self.mean_diag.prefix.text()
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
    def choose_dir(self):
        curr_dir = self.mean_diag.output_dir.text()
        dir_ = QFileDialog.getExistingDirectory(
            self, "Choose directory", curr_dir
        )
        self.mean_diag.output_dir.setText(dir_)

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
                data_shape = get_data_shape(filepath)
                nb_frame += data_shape[dataset][0]
            except IOError:
                self.add_info('Failed to open %s' % filepath)
                pass
        self.mean_diag.nb_frame.setText(str(nb_frame))

# file list / image view slots
    @pyqtSlot(QPoint)
    def show_menu(self, pos):
        menu = QMenu()
        item = self.file_list.currentItem()
        if not isinstance(item, QListWidgetItem):
            return
        filepath = item.data(1)
        ext = filepath.split('.')[-1]
        action_set_as_mask = menu.addAction('set as mask')
        action_select_and_load_dataset = menu.addAction(
            'select and load dataset'
        )
        action_calc_mean_std = menu.addAction('calculate mean/sigma')
        action_multiply_masks = menu.addAction('multiply masks')
        menu.addSeparator()
        action_del_file = menu.addAction('delete file(s)')
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        if action == action_set_as_mask:
            self.mask_file = filepath
            self.mask = read_image(filepath)
            self.status_params.param('mask file').setValue(filepath)
        elif action == action_select_and_load_dataset:
            if ext == 'npy':
                self.add_info(
                    'Unsupported file type for dataset selection: %s' % ext
                )
                return  # ignore npy file for dataset selection
            data_shape = get_data_shape(filepath)
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
        elif action == action_calc_mean_std:
            if ext == 'npy':
                self.add_info(
                    'Unsupported file type for mean calculation: %s' % ext
                )
                return  # ignore npy files
            combo_box = self.mean_diag.combo_box
            combo_box.clear()
            data_shape = get_data_shape(filepath)
            for dataset, shape in data_shape.items():
                combo_box.addItem(dataset)
            output_dir = os.path.join(self.workdir, 'mean')
            self.mean_diag.output_dir.setText(output_dir)
            self.mean_diag.progress_bar.setValue(0)
            self.mean_diag.exec_()
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
            mask = multiply_masks(mask_files)
            np.save(save_file, mask)
            self.add_info(
                'Making mask %s from %s' % (save_file, mask_files)
            )

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
            data_shape = get_data_shape(filepath)
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
            data_shape = get_data_shape(filepath)
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
        data_shape = get_data_shape(filepath)
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
        self.img = read_image(
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
                self.add_info('%d raw peaks found' % len(raw_peaks))
            valid_peaks = peaks_dict['valid']
            if valid_peaks is not None:
                self.add_info(
                    '%d peaks remaining after mask cleaning'
                    % len(peaks_dict['valid'])
                )
                self.peak_item.setData(
                    pos=valid_peaks + 0.5, symbol='x', size=self.peak_size,
                    pen='r', brush=(255, 255, 255, 0)
                )
            # refine peak position
            opt_peaks = peaks_dict['opt']
            if opt_peaks is not None:
                self.opt_peak_item.setData(
                    pos=opt_peaks + 0.5, symbol='+', size=self.peak_size,
                    pen='y', brush=(255, 255, 255, 0)
                )
            # filtering weak peak
            strong_peaks = peaks_dict['strong']
            if strong_peaks is not None:
                self.add_info('%d strong peaks' % (len(strong_peaks)))
                if len(strong_peaks) > 0:
                    self.strong_peak_item.setData(
                        pos=strong_peaks + 0.5,
                        symbol='o',
                        size=self.peak_size,
                        pen='g', brush=(255, 255, 255, 0)
                    )
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
    print(settings)
    app = QApplication(sys.argv)
    win = GUI(settings=settings)
    win.setWindowTitle('SFX Suite')
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
