"""
SFX-suite settings.
"""
import yaml


MODE = 'test'

if MODE == 'test':
    with open('conf/test.yml', 'r') as f:
        settings = yaml.load(f)