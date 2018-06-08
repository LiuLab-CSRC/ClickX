import os
from glob import glob
from functools import partial
import yaml

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QLineEdit, QComboBox, QSpinBox, \
    QDoubleSpinBox, QCheckBox, QFileDialog
from PyQt5.QtCore import pyqtSignal


class SettingDialog(QDialog):
    attribute_changed = pyqtSignal(tuple)

    def __init__(self):
        super(SettingDialog, self).__init__()
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/settings_diag.ui' % dir_, self)
        # general
        self.workDir.editingFinished.connect(
            partial(self.update_attribute,
                    attr='workdir', widget=self.workDir)
        )
        self.browseButton.pressed.connect(self.choose_workdir)
        self.jobEngine.currentIndexChanged.connect(
            partial(self.update_attribute,
                    attr='engine', widget=self.jobEngine)
        )
        # experiment
        self.photonEnergy.valueChanged.connect(
            partial(self.update_attribute,
                    attr='photon_energy', widget=self.photonEnergy)
        )
        self.detectorDistance.valueChanged.connect(
            partial(self.update_attribute,
                    attr='detector_distance', widget=self.detectorDistance)
        )
        self.pixelSize.valueChanged.connect(
            partial(self.update_attribute,
                    attr='pixel_size', widget=self.pixelSize)
        )
        self.imageWidth.valueChanged.connect(
            partial(self.update_attribute,
                    attr='image_width', widget=self.imageWidth)
        )
        self.imageHeight.valueChanged.connect(
            partial(self.update_attribute,
                    attr='image_height', widget=self.imageHeight)
        )
        self.centerX.valueChanged.connect(
            partial(self.update_attribute,
                    attr='center_x', widget=self.centerX)
        )
        self.centerY.valueChanged.connect(
            partial(self.update_attribute,
                    attr='center_y', widget=self.centerY)
        )
        # batch job
        self.compressRawData.stateChanged.connect(
            partial(self.update_attribute,
                    attr='compress_raw_data',
                    widget=self.compressRawData)
        )
        self.rawDataset.editingFinished.connect(
            partial(self.update_attribute,
                    attr='raw_dataset', widget=self.rawDataset)
        )
        self.compressedDataset.editingFinished.connect(
            partial(self.update_attribute,
                    attr='compressed_dataset', widget=self.compressedDataset)
        )
        self.hitTags.currentIndexChanged.connect(
            partial(self.update_attribute,
                    attr='curr_hit_tag', widget=self.hitTags)
        )
        self.minPeaks.valueChanged.connect(
            partial(self.update_attribute,
                    attr='min_peaks', widget=self.minPeaks)
        )
        self.cxiRawDataPath.editingFinished.connect(
            partial(self.update_attribute,
                    attr='cxi_raw_data_path',
                    widget=self.cxiRawDataPath)
        )
        self.cxiPeakInfoPath.editingFinished.connect(
            partial(self.update_attribute,
                    attr='cxi_peak_info_path',
                    widget=self.cxiPeakInfoPath)
        )
        self.cheetahDatasets.editingFinished.connect(
            partial(self.update_attribute,
                    attr='cheetah_datasets',
                    widget=self.cheetahDatasets)
        )
        self.jobPoolSize.valueChanged.connect(
            partial(self.update_attribute, attr='job_pool_size',
                    widget=self.jobPoolSize)
        )
        self.mpiBatchSize.valueChanged.connect(
            partial(self.update_attribute,
                    attr='mpi_batch_size',
                    widget=self.mpiBatchSize)
        )
        self.updatePeriod.valueChanged.connect(
            partial(self.update_attribute, attr='update_period',
                    widget=self.updatePeriod)
        )
        # other
        self.maxInfo.valueChanged.connect(
            partial(self.update_attribute,
                    attr='max_info', widget=self.maxInfo)
        )

    def update(self, **kwargs):
        # general
        if 'workdir' in kwargs:
            self.workDir.setText(kwargs['workdir'])
        if 'engines' in kwargs:
            self.jobEngine.clear()
            for engine in kwargs['engines']:
                self.jobEngine.addItem(engine)
        if 'engine' in kwargs:
            engines = []
            for i in range(self.jobEngine.count()):
                engines.append(self.jobEngine.itemText(i))
            engine_id = engines.index(kwargs['engine'])
            self.jobEngine.setCurrentIndex(engine_id)
        # experiment
        if 'photon_energy' in kwargs:
            self.photonEnergy.setValue(kwargs['photon_energy'])
        if 'detector_distance' in kwargs:
            self.detectorDistance.setValue(kwargs['detector_distance'])
        if 'pixel_size' in kwargs:
            self.pixelSize.setValue(kwargs['pixel_size'])
        if 'image_width' in kwargs:
            self.imageWidth.setValue(kwargs['image_width'])
        if 'image_height' in kwargs:
            self.imageHeight.setValue(kwargs['image_height'])
        if 'center_x' in kwargs:
            self.centerX.setValue(kwargs['center_x'])
        if 'center_y' in kwargs:
            self.centerY.setValue(kwargs['center_y'])
        # batch job
        if 'compress_raw_data' in kwargs:
            self.compressRawData.setChecked(kwargs['compress_raw_data'])
        if 'raw_dataset' in kwargs:
            self.rawDataset.setText(kwargs['raw_dataset'])
        if 'compressed_dataset' in kwargs:
            self.compressedDataset.setText(kwargs['compressed_dataset'])
        if 'mpi_batch_size' in kwargs:
            self.mpiBatchSize.setValue(kwargs['mpi_batch_size'])
        if 'min_peaks' in kwargs:
            self.minPeaks.setValue(kwargs['min_peaks'])
        if 'hit_conf_tags' in kwargs:
            self.hitTags.clear()
            for tag in kwargs['hit_conf_tags']:
                self.hitTags.addItem(tag)
        if 'curr_hit_tag' in kwargs:
            if kwargs['curr_hit_tag'] is not None:
                hit_conf_tags = []
                for i in range(self.hitTags.count()):
                    hit_conf_tags.append(self.hitTags.itemText(i))
                tag_id = hit_conf_tags.index(kwargs['curr_hit_tag'])
            else:
                tag_id = 0
            self.hitTags.setCurrentIndex(tag_id)
        if 'cxi_raw_data_path' in kwargs:
            self.cxiRawDataPath.setText(kwargs['cxi_raw_data_path'])
        if 'cxi_peak_info_path' in kwargs:
            self.cxiPeakInfoPath.setText(kwargs['cxi_peak_info_path'])
        if 'cheetah_datasets' in kwargs:
            self.cheetahDatasets.setText(kwargs['cheetah_datasets'])
        if 'job_pool_size' in kwargs:
            self.jobPoolSize.setValue(kwargs['job_pool_size'])
        if 'update_period' in kwargs:
            self.updatePeriod.setValue(kwargs['update_period'])
        # other
        if 'max_info' in kwargs:
            self.maxInfo.setValue(kwargs['max_info'])

    def update_attribute(self, attr, widget):
        if isinstance(widget, QLineEdit):
            value = widget.text()
        elif isinstance(widget, QComboBox):
            value = widget.currentText()
        elif isinstance(widget, QSpinBox):
            value = widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            value = widget.value()
        elif isinstance(widget, QCheckBox):
            value = widget.isChecked()
        if attr == 'compress_raw_data':
            if widget.isChecked():
                self.rawDataset.setEnabled(True)
                self.compressedDataset.setEnabled(True)
            else:
                self.rawDataset.setEnabled(False)
                self.compressedDataset.setEnabled(False)
        self.attribute_changed.emit((attr, value))

    def get_checked_table_columns(self):
        columns = []
        for i in range(self.tableColumns.count()):
            item = self.tableColumns.itemAt(i).widget()
            if item.isChecked():
                columns.append(item.text())
        return columns

    def choose_workdir(self):
        dir_ = QFileDialog.getExistingDirectory(
            self, "Choose directory", self.workDir.text())
        if len(dir_) == 0:
            return
        self.workDir.setText(dir_)


class Settings(object):
    setting_file = '.config.yml'
    saved_attrs = (
        'workdir', 'engine', 'photon_energy', 'detector_distance',
        'pixel_size', 'image_width', 'image_height', 'center_x',
        'center_y', 'compress_raw_data', 'raw_dataset',
        'compressed_dataset', 'mpi_batch_size',
        'cxi_raw_data_path', 'cxi_peak_info_path', 'cheetah_datasets',
        'min_peaks', 'max_info', 'job_pool_size', 'update_period',
        'curr_hit_tag'
    )

    def __init__(self, setting_diag):
        super(Settings, self).__init__()
        self.setting_diag = setting_diag
        self.workdir = None
        self.engines = None  # available engines
        self.engine = None  # current engine
        self.hit_conf_tags = None
        self.curr_hit_tag = None
        self.photon_energy = None
        self.detector_distance = None
        self.pixel_size = None
        self.image_width = None
        self.image_height = None
        self.center_x = None
        self.center_y = None
        self.compress_raw_data = None
        self.raw_dataset = None
        self.compressed_dataset = None
        self.mpi_batch_size = None
        self.cxi_raw_data_path = None
        self.cxi_peak_info_path = None
        self.cheetah_datasets = None  # extra datasets from cheetah data
        self.job_pool_size = None
        self.update_period = None
        self.min_peaks = None
        self.max_info = None
        self.load_settings()
        # signal/slots
        self.setting_diag.attribute_changed.connect(self.update)

    def load_settings(self):
        settings = None
        if os.path.exists(self.setting_file):
            with open(self.setting_file) as f:
                settings = yaml.load(f)
        if settings is None:
            settings = {}
        self.update(workdir=settings.get('workdir', os.getcwd()))
        self.update(engines=get_all_engines())
        self.update(engine=settings.get('engine', 'local'))
        self.update(photon_energy=settings.get('photon_energy', 9000))
        self.update(detector_distance=settings.get('detector_distance', 100))
        self.update(pixel_size=settings.get('pixel_size', 100))
        self.update(image_width=settings.get('image_width', 1000))
        self.update(image_height=settings.get('image_height', 1000))
        self.update(center_x=settings.get('center_x', 500))
        self.update(center_y=settings.get('center_y', 500))
        self.update(compress_raw_data=settings.get('compress_raw_data', True))
        self.update(raw_dataset=settings.get('raw_dataset', 'data'))
        self.update(compressed_dataset=settings.get(
            'compressed_dataset', 'data'))
        self.update(mpi_batch_size=settings.get(
            'mpi_batch_size', 5))
        self.update(cxi_raw_data_path=settings.get(
            'cxi_raw_data_path', 'data'))
        self.update(cxi_peak_info_path=settings.get(
            'cxi_peak_info_path', 'peak_info'))
        self.update(cheetah_datasets=settings.get(
            'cheetah_datasets', ''))
        self.update(job_pool_size=settings.get('job_pool_size', 4))
        self.update(update_period=settings.get('update_period', 2))
        self.update(min_peaks=settings.get('min_peaks', 20))
        self.update(max_info=settings.get('max_info', 1000))
        self.update(hit_conf_tags=get_all_hit_tags(self.workdir))
        self.update(curr_hit_tag=settings.get('curr_hit_tag', None))

    def save_settings(self):
        settings = {attr: getattr(self, attr) for attr in self.saved_attrs}
        with open(self.setting_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

    def update(self, *args, **kwargs):
        for arg in args:
            setattr(self, arg[0], arg[1])
            if arg[0] == 'workdir':
                self.update(hit_conf_tags=get_all_hit_tags(self.workdir))
                self.update(curr_hit_tag=self.curr_hit_tag)
        for key, value in kwargs.items():
            setattr(self, key, value)
            if key == 'workdir':
                self.update(hit_conf_tags=get_all_hit_tags(self.workdir))
                self.update(curr_hit_tag=self.curr_hit_tag)
        self.setting_diag.update(**kwargs)

    def __str__(self):
        s = ''
        for attr in self.saved_attrs:
            s += '%s: %s\n' % (attr, getattr(self, attr))
        return s


def get_all_engines():
    dir_ = os.path.abspath(os.path.dirname(__file__))
    engines = os.listdir('%s/engines' % dir_)
    # remove fake engines if start with .
    engines = [engine for engine in engines if not engine[0] == '.']
    return engines


def get_all_hit_tags(workdir):
    conf_dir = '%s/conf' % workdir
    hit_files = glob('%s/*.yml' % conf_dir)
    hit_tags = [hit_file.split('/')[-1].split('.')[0]
                for hit_file in hit_files]
    return hit_tags
