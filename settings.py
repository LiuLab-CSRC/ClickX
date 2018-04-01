"""
SFX-suite settings.
"""
import os


class Settings():
    def __init__(self, settings_dict={}):
        # main window
        self.workdir = settings_dict.get('work dir', os.path.dirname(__file__))
        self.peak_size = settings_dict.get('peak size', 10)
        self.dataset_def = settings_dict.get('default dataset', '')
        self.max_info = settings_dict.get('max info', 1000)

        # compression
        self.raw_dataset = settings_dict.get('raw dataset', None)
        self.comp_dtype = settings_dict.get('compressed dtype', 'auto')
        self.comp_size = settings_dict.get('compressed size', '1000')
        self.comp_dataset = settings_dict.get('compressed dataset', None)

        # job window
        self.header_labels = settings_dict.get(
            'header labels',
            ['job', 'compression', 'compression ratio', 'raw frames',
             'tag', 'hit finding', 'processed frames', 'processed hits', 'hit rate']
        )

        # scripts
        self.script_suffix = settings_dict.get('script suffix', 'local')

    def __str__(self):
        attrs = dir(self)
        s = ''
        for attr in attrs:
            if attr[:2] != '__':
                s += '%s: %s\n' % (attr, getattr(self, attr))
        return s
