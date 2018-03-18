import sys
import os
from functools import partial

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QListWidgetItem, QDialog, QMenu, QFileDialog
from PyQt5.uic import loadUi

from util import *
import yaml

PEAK_SIZE = 12


class GUI(QMainWindow):
    def __init__(self, *args):
        super(GUI, self).__init__(*args)

        # setup layout
        loadUi('gui.ui', self)
        self.win2 = QDialog()
        self.win2.setWindowFlags(Qt.WindowStaysOnTopHint)
        loadUi('win2.ui', self.win2)
        self.dataset_diag = QDialog()
        loadUi('dataset_diag.ui', self.dataset_diag)
        self.splitter.setSizes(
            [0.7 * self.width(), 0.0 * self.width(), 0.3 * self.width()])
        self.image_view_2.hide()
        self.setAcceptDrops(True)

        self.accepted_file_types = ('h5', 'npy', 'cxi')
        self.mask_file = None
        self.file = None
        self.h5_obj = None
        self.h5_dataset = None
        self.h5_dataset_def = ''
        self.frame = 0
        self.show_view2 = False
        self.show_win2 = False
        self.mask_on = False
        self.refine_on = False
        self.hit_finding_on = False
        self.gaussian_sigma = 1
        self.min_snr = 4.
        self.min_peak_num = 0
        self.max_peak_num = 500
        self.min_intensity = 0.
        self.min_gradient = 0.
        self.min_distance = 10

        self.img = None
        self.img2 = None
        self.mask = None
        self.peak_item = None
        self.opt_peak_item = None
        self.strong_peak_item = None

        # setup parameter tree
        params_list = [
            {
                'name': 'File Info', 'type': 'group', 'children': [
                    {'name': 'filepath', 'type': 'str', 'readonly': True},
                    {'name': 'dataset', 'type': 'str', 'readonly': True},
                    {'name': 'image num', 'type': 'str', 'readonly': True},
                    {'name': 'mask file', 'type': 'str', 'readonly': True},
                ]
            },
            {
                'name': 'Basic Operation', 'type': 'group', 'children': [
                    {'name': 'frame', 'type': 'int', 'value': self.frame},
                    {'name': 'hit finding on', 'type': 'bool',
                        'value': self.hit_finding_on},
                    {'name': 'mask on', 'type': 'bool',
                        'value': self.mask_on},
                    {'name': 'refine on', 'type': 'bool',
                        'value': self.refine_on},
                    {'name': 'show view2', 'type': 'bool',
                        'value': self.show_view2},
                    {'name': 'show win2', 'type': 'bool',
                        'value': self.show_win2},
                    {'name': 'save hit finding conf file', 'type': 'action'},
                    {'name': 'load hit finding conf file', 'type': 'action'}
                ]
            },
            {
                'name': 'Hit Finder Parameters', 'type': 'group', 'children': [
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
            },
        ]
        self.params = Parameter.create(
            name='params', type='group', children=params_list)
        self.hit_finder_tree.setParameters(self.params, showTop=False)

        # signal and slot
        self.image_view.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=1))
        self.image_view_2.scene.sigMouseMoved.connect(
            partial(self.mouse_moved, flag=2))
        self.file_list.itemDoubleClicked.connect(self.load_file)
        self.file_list.customContextMenuRequested.connect(
            self.show_menu)
        self.line_edit.returnPressed.connect(self.add_file)
        self.params.param('Basic Operation', 'frame').sigValueChanged.connect(
            self.change_frame)
        self.params.param(
            'Basic Operation', 'hit finding on').sigValueChanged.connect(
            self.change_hit_finding)
        self.params.param(
            'Basic Operation', 'show view2').sigValueChanged.connect(
            self.change_show_view2)
        self.params.param(
            'Basic Operation', 'show win2').sigValueChanged.connect(
            self.change_show_win2)
        self.params.param(
            'Basic Operation', 'mask on').sigValueChanged.connect(
            self.apply_mask)
        self.params.param(
            'Basic Operation', 'refine on').sigValueChanged.connect(
            self.apply_refine)
        self.params.param(
            'Basic Operation', 'save hit finding conf file'
        ).sigActivated.connect(
            self.save_conf)
        self.params.param(
            'Basic Operation', 'load hit finding conf file'
        ).sigActivated.connect(
            self.load_conf)
        self.params.param(
            'Hit Finder Parameters', 'gaussian filter sigma'
        ).sigValueChanged.connect(
            self.change_gaussian_sigma)
        self.params.param(
            'Hit Finder Parameters', 'min peak num').sigValueChanged.connect(
            self.change_min_peak_num)
        self.params.param(
            'Hit Finder Parameters', 'max peak num').sigValueChanged.connect(
            self.change_max_peak_num)
        self.params.param(
            'Hit Finder Parameters', 'min gradient').sigValueChanged.connect(
            self.change_min_gradient)
        self.params.param(
            'Hit Finder Parameters', 'min distance').sigValueChanged.connect(
            self.change_min_distance)
        self.params.param(
            'Hit Finder Parameters', 'min snr').sigValueChanged.connect(
            self.change_min_snr)

    @pyqtSlot(object)
    def mouse_moved(self, pos, flag=None):
        if self.file is None:
            return
        if flag == 1:  # in image_view
            mouse_point = self.image_view.view.mapToView(pos)
        elif flag == 2:  # in image_view_2
            mouse_point = self.image_view_2.view.mapToView(pos)
        else:
            return
        x, y = int(mouse_point.x()), int(mouse_point.y())
        if 0 <= x < self.img.shape[0] and 0 <= y < self.img.shape[1]:
            self.statusbar.showMessage(
                "x:%d y:%d I1:%.2E I2: %.2E" %
                (x, y, self.img[x, y], self.img2[x, y]), 5000)
        else:
            return
        if self.show_win2:  # show data inspector
            # out of bound check
            if x - 3 < 0 or x + 4 > self.img.shape[0]:
                return
            elif y - 3 < 0 or y + 4 > self.img.shape[1]:
                return
            # calculate snr
            crop = self.img[x - 3:x + 4, y - 3:y + 4]
            crop = np.reshape(crop, (-1, 7, 7))
            snr = calc_snr(crop)
            self.win2.snr_label.setText('SNR@(%d, %d):' % (x, y))
            self.win2.snr_value.setText('%.1f' % (snr))
            # set table values
            for i in range(5):
                for j in range(5):
                    v1 = self.img[x + i - 2, y + j - 2]
                    if self.show_view2:
                        v2 = self.img2[x + i - 2, y + j - 2]
                        item = QtGui.QTableWidgetItem('%d\n%d' % (v1, v2))
                    else:
                        item = QtGui.QTableWidgetItem('%d' % v1)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.win2.data_table.setItem(j, i, item)

    @pyqtSlot(QPoint)
    def show_menu(self, pos):
        menu = QMenu()
        item = self.file_list.currentItem()
        if not isinstance(item, QListWidgetItem):
            return
        filepath = item.text()
        set_as_mask = menu.addAction('set as mask')
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        if action == set_as_mask:
            self.mask_file = filepath
            self.mask = read_image(filepath)
            self.params.param('File Info', 'mask file').setValue(filepath)

    @pyqtSlot(object, object)
    def apply_mask(self, _, mask_on):
        self.mask_on = mask_on
        self.update_display()

    @pyqtSlot(object, object)
    def apply_refine(self, _, refine_on):
        self.refine_on = refine_on
        self.update_display()

    @pyqtSlot(object, object)
    def change_show_view2(self, _, show_view2):
        self.show_view2 = show_view2
        if self.show_view2:
            self.image_view_2.show()
        else:
            self.image_view_2.hide()

    @pyqtSlot(object, object)
    def change_show_win2(self, _, show_win2):
        self.show_win2 = show_win2
        if self.show_win2:
            self.win2.show()
        else:
            self.win2.hide()

    @pyqtSlot(object, object)
    def change_hit_finding(self, _, hit_finding_on):
        self.hit_finding_on = hit_finding_on
        self.update_display()

    @pyqtSlot(object)
    def save_conf(self, _):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Hit Finding Conf File", "", "Yaml Files (*.yml)")
        conf_dict = {
            'dataset': self.h5_dataset,
            'mask file': self.mask_file,
            'refine on': self.refine_on,
            'gaussian filter sigma': self.gaussian_sigma,
            'min peak num': self.min_peak_num,
            'max peak num': self.max_peak_num,
            'min gradient': self.min_gradient,
            'min distance': self.min_distance,
            'min snr': self.min_snr,
        }
        with open(filepath, 'w') as f:
            yaml.dump(conf_dict, f, default_flow_style=False)

    @pyqtSlot(object)
    def load_conf(self, _):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Hit Finding Conf File", "", "Yaml Files (*.yml)")
        with open(filepath, 'r') as f:
            conf_dict = yaml.load(f)
        if 'dataset' in conf_dict.keys():
            self.h5_dataset_def = self.h5_dataset = conf_dict['dataset']
        if 'mask file' in conf_dict.keys():
            self.mask_file = conf_dict['mask file']
            self.params.param(
                'File Info', 'mask file').setValue(self.mask_file)
            self.mask = read_image(self.mask_file)
        if 'refine on' in conf_dict.keys():
            self.refine_on = conf_dict['refine on']
            self.params.param('Basic Operation', 'refine on').setValue(
                self.refine_on)
        if 'gaussian filter sigma' in conf_dict.keys():
            self.gaussian_sigma = conf_dict['gaussian filter sigma']
            self.params.param(
                'Hit Finder Parameters', 'gaussian filter sigma'
            ).setValue(self.gaussian_sigma)
        if 'min peak num' in conf_dict.keys():
            self.min_peak_num = conf_dict['min peak num']
            self.params.param(
                'Hit Finder Parameters', 'min peak num'
            ).setValue(self.min_peak_num)
        if 'max peak num' in conf_dict.keys():
            self.max_peak_num = conf_dict['max peak num']
            self.params.param(
                'Hit Finder Parameters', 'max peak num'
            ).setValue(self.max_peak_num)
        if 'min gradient' in conf_dict.keys():
            self.min_gradient = conf_dict['min gradient']
            self.params.param(
                'Hit Finder Parameters', 'min gradient'
            ).setValue(self.min_gradient)
        if 'min distance' in conf_dict.keys():
            self.min_distance = conf_dict['min distance']
            self.params.param(
                'Hit Finder Parameters', 'min distance'
            ).setValue(self.min_distance)
        if 'min snr' in conf_dict.keys():
            self.min_snr = conf_dict['min snr']
            self.params.param(
                'Hit Finder Parameters', 'min snr'
            ).setValue(self.min_snr)

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

    @pyqtSlot()
    def add_file(self):
        self.maybe_add_file(self.line_edit.text())

    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > self.nb_frame - 1:
            frame = self.nb_frame - 1
        self.frame = frame
        self.params.param('Basic Operation', 'frame').setValue(self.frame)
        self.update_display()

    def choose_dataset(self, filepath):
        combo_box = self.dataset_diag.combo_box
        combo_box.clear()
        data_info = get_h5_info(filepath)
        for i in range(len(data_info)):
            combo_box.addItem(
                '%s  %s' %
                (data_info[i]['key'], data_info[i]['shape']),
                userData=data_info[i]['key'])
        self.dataset_diag.show()
        if self.dataset_diag.exec_() == QDialog.Accepted:
            id_select = combo_box.currentIndex()
            h5_dataset = combo_box.itemData(id_select)
            if self.dataset_diag.check_box.isChecked():
                self.h5_dataset_def = h5_dataset
            return h5_dataset
        else:
            return ''

    @pyqtSlot('QListWidgetItem*')
    def load_file(self, file_item):
        print('loading %s' % file_item.text())
        filepath = file_item.text()
        ext = QtCore.QFileInfo(filepath).suffix()
        if ext == 'npy':
            self.file = filepath
            self.nb_frame = 1
        elif ext in ('h5', 'cxi'):
            h5_obj = h5py.File(filepath, 'r')
            if self.h5_dataset_def not in h5_obj:  # check default dataset
                h5_dataset = self.choose_dataset(filepath)
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
        self.params.param('File Info', 'filepath').setValue(filepath)
        self.params.param('File Info', 'dataset').setValue(self.h5_dataset)
        self.params.param('File Info', 'image num').setValue(self.nb_frame)
        self.update_display()

    def update_display(self):
        if self.file is None:
            return
        self.img = read_image(
            self.file, frame=self.frame,
            h5_obj=self.h5_obj, h5_dataset=self.h5_dataset
        ).astype(np.float32)
        self.img[0, -1] = 5000.
        self.image_view.setImage(self.img, autoRange=False,
                                 autoLevels=False, autoHistogramRange=False)
        # smooth the image
        if self.gaussian_sigma > 0:
            self.img2 = gaussian_filter(self.img, self.gaussian_sigma)
        else:
            self.img2 = self.img.copy()
        # calculate gradient
        grad = np.gradient(self.img2.astype(np.float32))
        self.img2 = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
        self.image_view_2.setImage(self.img2, autoRange=False,
                                   autoLevels=False, autoHistogramRange=False)

        if self.peak_item is not None:
            self.peak_item.clear()
        if self.opt_peak_item is not None:
            self.opt_peak_item.clear()
        if self.strong_peak_item is not None:
            self.strong_peak_item.clear()
        if self.hit_finding_on:
            peaks = peak_local_max(
                self.img2,
                min_distance=int(round((self.min_distance - 1.) / 2.)),
                threshold_abs=self.min_gradient, num_peaks=self.max_peak_num)
            print('%d peaks found' % len(peaks))
            if len(peaks) == 0:
                return
            # remove peaks in mask area
            if self.mask_on and self.mask is not None:
                valid_peak_ids = []
                for i in range(peaks.shape[0]):
                    peak = np.round(peaks[i].astype(np.int))
                    if self.mask[peak[0], peak[1]] == 1:
                        valid_peak_ids.append(i)
                peaks = peaks[valid_peak_ids]
                print('%d peaks remaining after mask cleaning' % len(peaks))
            if len(peaks) == 0:
                return
            self.peak_item = pg.ScatterPlotItem(
                pos=peaks + 0.5, symbol='x', size=PEAK_SIZE,
                pen='r', brush=(255, 255, 255, 0))
            self.image_view.getView().addItem(self.peak_item)
            # refine peak postion
            if self.refine_on:
                opt_peaks = refine_peaks(self.img, peaks)
                self.opt_peak_item = pg.ScatterPlotItem(
                    pos=opt_peaks + 0.5, symbol='+', size=PEAK_SIZE,
                    pen='y', brush=(255, 255, 255, 0))
                self.image_view.getView().addItem(self.opt_peak_item)

                # filtering peaks using snr threshold
                crops = []
                for i in range(len(opt_peaks)):
                    x, y = np.round(opt_peaks[i]).astype(np.int)
                    if x - 3 < 0 or x + 4 > self.img.shape[0]:
                        continue
                    elif y - 3 < 0 or y + 4 > self.img.shape[1]:
                        continue
                    crops.append(self.img[x - 3:x + 4, y - 3:y + 4])
                crops = np.array(crops)
                crops = np.reshape(crops, (-1, 7, 7))

                snr = calc_snr(crops)
                strong_peaks = opt_peaks[snr >= self.min_snr]
                print('%d strong peaks' % (len(strong_peaks)))
                if len(strong_peaks) > 0:
                    self.strong_peak_item = pg.ScatterPlotItem(
                        pos=strong_peaks + 0.5, symbol='o', size=PEAK_SIZE,
                        pen='g', brush=(255, 255, 255, 0))
                    self.image_view.getView().addItem(self.strong_peak_item)

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
        ext = drop_file.split('.')[-1]
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
