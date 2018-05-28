import os
from functools import partial
import yaml

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QLineEdit, QComboBox, QSpinBox, \
    QDoubleSpinBox, QCheckBox, QFileDialog
from PyQt5.QtCore import pyqtSignal


class SettingDialog(QDialog):
    attribute_changed = pyqtSignal(tuple)
    compressed_datatypes = ('auto', 'int16', 'int32', 'int64',
                            'float32', 'float64')

    def __init__(self):
        super(SettingDialog, self).__init__()
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/settings_diag.ui' % dir_, self)
        self.workDir.editingFinished.connect(
            partial(self.update_attribute,
                    attr='workdir', widget=self.workDir)
        )
        self.browseButton.pressed.connect(self.choose_workdir)
        self.jobEngine.currentIndexChanged.connect(
            partial(self.update_attribute,
                    attr='engine', widget=self.jobEngine)
        )
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
        self.rawDataset.editingFinished.connect(
            partial(self.update_attribute,
                    attr='raw_dataset', widget=self.rawDataset)
        )
        self.compressedDataset.editingFinished.connect(
            partial(self.update_attribute,
                    attr='compressed_dataset', widget=self.compressedDataset)
        )
        self.compressedDatatype.currentIndexChanged.connect(
            partial(self.update_attribute,
                    attr='compressed_datatype', widget=self.compressedDatatype)
        )
        self.compressedBatchSize.valueChanged.connect(
            partial(self.update_attribute,
                    attr='compressed_batch_size',
                    widget=self.compressedBatchSize)
        )
        self.compressionProgress.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.compressionProgress)
        )
        self.compressionRatio.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.compressionRatio)
        )
        self.rawFrames.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.rawFrames)
        )
        self.hitFindingProgress.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.hitFindingProgress)
        )
        self.processedFrames.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.processedFrames)
        )
        self.processedHits.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.processedHits)
        )
        self.hitRate.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.hitRate)
        )
        self.peak2cxiProgress.stateChanged.connect(
            partial(self.update_attribute, attr='table_columns',
                    widget=self.peak2cxiProgress)
        )
        self.jobPoolSize.valueChanged.connect(
            partial(self.update_attribute, attr='job_pool_size',
                    widget=self.jobPoolSize)
        )
        self.maxInfo.valueChanged.connect(
            partial(self.update_attribute,
                    attr='max_info', widget=self.maxInfo)
        )
        for datatype in self.compressed_datatypes:
            self.compressedDatatype.addItem(datatype)

    def update(self, **kwargs):
        if 'workdir' in kwargs:
            self.workDir.setText(kwargs['workdir'])
        if 'engines' in kwargs:
            for engine in kwargs['engines']:
                self.jobEngine.addItem(engine)
        if 'engine' in kwargs:
            engines = []
            for i in range(self.jobEngine.count()):
                engines.append(self.jobEngine.itemText(i))
            engine_id = engines.index(kwargs['engine'])
            self.jobEngine.setCurrentIndex(engine_id)
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
        if 'raw_dataset' in kwargs:
            self.rawDataset.setText(kwargs['raw_dataset'])
        if 'compressed_dataset' in kwargs:
            self.compressedDataset.setText(kwargs['compressed_dataset'])
        if 'compressed_datatype' in kwargs:
            datatype_id = self.compressed_datatypes.index(
                kwargs['compressed_datatype'])
            self.compressedDatatype.setCurrentIndex(datatype_id)
        if 'compressed_batch_size' in kwargs:
            self.compressedBatchSize.setValue(kwargs['compressed_batch_size'])
        if 'table_columns' in kwargs:
            if 'job id' in kwargs['table_columns']:
                self.jobId.setChecked(True)
            else:
                self.jobId.setChecked(False)
            if 'compression progress' in kwargs['table_columns']:
                self.compressionProgress.setChecked(True)
            else:
                self.compressionProgress.setChecked(False)
            if 'compression ratio' in kwargs['table_columns']:
                self.compressionRatio.setChecked(True)
            else:
                self.compressionRatio.setChecked(False)
            if 'raw frames' in kwargs['table_columns']:
                self.rawFrames.setChecked(True)
            else:
                self.rawFrames.setChecked(False)
            if 'tag id' in kwargs['table_columns']:
                self.tagId.setChecked(True)
            else:
                self.tagId.setChecked(False)
            if 'hit finding progress' in kwargs['table_columns']:
                self.hitFindingProgress.setChecked(True)
            else:
                self.hitFindingProgress.setChecked(False)
            if 'processed frames' in kwargs['table_columns']:
                self.processedFrames.setChecked(True)
            else:
                self.processedFrames.setChecked(False)
            if 'processed hits' in kwargs['table_columns']:
                self.processedHits.setChecked(True)
            else:
                self.processedHits.setChecked(False)
            if 'hit rate' in kwargs['table_columns']:
                self.hitRate.setChecked(True)
            else:
                self.hitRate.setChecked(False)
            if 'peak2cxi progress' in kwargs['table_columns']:
                self.peak2cxiProgress.setChecked(True)
            else:
                self.peak2cxiProgress.setChecked(False)
        if 'job_pool_size' in kwargs:
            self.jobPoolSize.setValue(kwargs['job_pool_size'])
        if 'min_peaks' in kwargs:
            self.minPeaks.setValue(kwargs['min_peaks'])
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
            value = self.get_checked_table_columns()
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
    saved_attrs = ('workdir', 'engine', 'photon_energy', 'detector_distance',
                   'pixel_size', 'image_width', 'image_height', 'center_x',
                   'center_y', 'raw_dataset', 'compressed_dataset',
                   'compressed_datatype', 'compressed_batch_size',
                   'table_columns', 'min_peaks', 'max_info', 'job_pool_size')

    def __init__(self, setting_diag):
        super(Settings, self).__init__()
        self.setting_diag = setting_diag
        self.workdir = None
        self.engines = None  # available engines
        self.engine = None  # current engine
        self.photon_energy = None
        self.detector_distance = None
        self.pixel_size = None
        self.image_width = None
        self.image_height = None
        self.center_x = None
        self.center_y = None
        self.raw_dataset = None
        self.compressed_dataset = None
        self.compressed_datatype = None
        self.compressed_batch_size = None
        self.table_columns = None
        self.job_pool_size = None
        self.min_peaks = None
        self.max_info = None
        self.update(engines=get_all_engines())
        self.load_settings()
        # signal/slots
        self.setting_diag.attribute_changed.connect(self.update)

    def load_settings(self):
        settings = None
        all_columns = ('job id', 'compression progress', 'compression ratio',
                       'raw frames', 'tag id', 'hit finding progress',
                       'processed frames', 'processed hits', 'hit rate',
                       'peak2cxi progress')
        if os.path.exists(self.setting_file):
            with open(self.setting_file) as f:
                settings = yaml.load(f)
        if settings is None:
            settings = {}
        self.update(workdir=settings.get('workdir', os.getcwd()))
        self.update(engine=settings.get('engine', 'local'))
        self.update(photon_energy=settings.get('photon_energy', 9000))
        self.update(detector_distance=settings.get('detector_distance', 100))
        self.update(pixel_size=settings.get('pixel_size', 100))
        self.update(image_width=settings.get('image_width', 1000))
        self.update(image_height=settings.get('image_height', 1000))
        self.update(center_x=settings.get('center_x', 500))
        self.update(center_y=settings.get('center_y', 500))
        self.update(raw_dataset=settings.get('raw_dataset', 'data'))
        self.update(compressed_dataset=settings.get(
            'compressed_dataset', 'data'))
        self.update(compressed_datatype=settings.get(
            'compressed_datatype', 'auto'))
        self.update(compressed_batch_size=settings.get(
            'compressed_batch_size', 200))
        self.update(table_columns=settings.get('table_columns', all_columns))
        self.update(job_pool_size=settings.get('job_pool_size', 4))
        self.update(min_peaks=settings.get('min_peaks', 20))
        self.update(max_info=settings.get('max_info', 1000))

    def save_settings(self):
        settings = {attr: getattr(self, attr) for attr in self.saved_attrs}
        with open(self.setting_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

    def update(self, *args, **kwargs):
        for arg in args:
            setattr(self, arg[0], arg[1])
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.setting_diag.update(**kwargs)

    def __str__(self):
        s = ''
        for attr in self.saved_attrs:
            s += '%s: %s\n' % (attr, getattr(self, attr))
        return s


def get_all_engines():
    dir_ = os.path.abspath(os.path.dirname(__file__))
    engines = os.listdir('%s/engines' % dir_)
    return engines
