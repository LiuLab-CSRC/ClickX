"""
SFX-suite settings.
"""
import os
import numpy as np


class Settings():
    def __init__(self, settings_dict={}):
        # main window
        self.workdir = settings_dict.get('work dir', os.path.dirname(__file__))
        self.peak_size = settings_dict.get('peak size', 10)
        self.dataset_def = settings_dict.get('default dataset', '')
        self.max_info = settings_dict.get('max info', 1000)
        self.min_peak = settings_dict.get('min peak', 20)

        # compression
        self.raw_dataset = settings_dict.get('raw dataset', None)
        self.comp_dtype = settings_dict.get('compressed dtype', 'auto')
        self.comp_size = settings_dict.get('compressed size', '1000')
        self.comp_dataset = settings_dict.get('compressed dataset', None)

        # job window
        self.header_labels = settings_dict.get(
            'header labels',
            [
                'job',
                'compression',
                'compression ratio',
                'raw frames',
                'tag',
                'hit finding',
                'processed frames',
                'processed hits',
                'hit rate',
                'peak2cxi'
            ]
        )

        # scripts
        self.script_suffix = settings_dict.get('script suffix', 'local')

        # powder fit
        self.max_peaks = settings_dict.get('max peaks', 1000)
        self.width = settings_dict.get('width', 1000)
        self.height = settings_dict.get('height', 1000)
        cx = settings_dict.get('center x', 500)
        cy = settings_dict.get('center y', 500)
        self.center = np.array([cx, cy])
        self.eps = settings_dict.get('eps', 5.0)
        self.min_samples = settings_dict.get('min samples', 10)
        self.tol = settings_dict.get('outlier tol', 2.)

        # experiment parameters
        self.photon_energy = settings_dict.get('photon energy', 9000)
        self.detector_distance = settings_dict.get('detector distance', 100)
        self.pixel_size = settings_dict.get('pixel size', 100)

    def __str__(self):
        attrs = dir(self)
        s = ''
        for attr in attrs:
            if attr[:2] != '__':
                s += '%s: %s\n' % (attr, getattr(self, attr))
        return s
