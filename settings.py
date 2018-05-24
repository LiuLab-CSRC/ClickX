import os
import yaml
import types
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog


class SettingDialog(QDialog):
    def __init__(self):
        super(SettingDialog, self).__init__()
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/settings_diag.ui' % dir_, self)


class Settings(object):
    setting_file = '.config.yml'
    saved_attrs = ('workdir',)

    def __init__(self, setting_diag):
        super(Settings, self).__init__()
        self.setting_diag = setting_diag
        self.workdir = None
        self.job_engine = None
        self.load_settings()

    def load_settings(self):
        settings = None
        if os.path.exists(self.setting_file):
            with open(self.setting_file) as f:
                settings = yaml.load(f)
        if settings is None:
            settings = {}
        # self.workdir = settings.get('workdir', os.getcwd())
        self.update(workdir=settings.get('workdir', os.getcwd()))

    def save_settings(self):
        settings = {attr: getattr(self, attr) for attr in self.saved_attrs}
        with open(self.setting_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

    def update(self, **kwargs):
        dialog = self.setting_diag
        if 'workdir' in kwargs:
            self.workdir = kwargs['workdir']
            dialog.workDirLine.setText(self.workdir)


# def get_all_attrs(instance):
#     attrs = []
#     for attr in dir(instance):
#         if attr[:2] == '__':  # skip private attributes
#             continue
#         if isinstance(getattr(instance, attr), types.MethodType):  # methods
#             continue
#         attrs.append(attr)
#     return attrs
