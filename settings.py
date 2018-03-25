"""
SFX-suite settings.
"""
import yaml


def get_settings(conf_file=None):
    if conf_file is None:
        with open('conf/test.yml', 'r') as f:
            return yaml.load(f)
    else:
        with open(conf_file, 'r') as f:
            return yaml.load(f)