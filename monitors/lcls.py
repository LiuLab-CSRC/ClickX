#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Monitor raw data and generate list file at LCLS.
Usage:
    lcls.py <data_dir> <lst_dir> [options]

Options:
    -h --help                   Show this screen.
    --version                   Show version.
    --only-once                 Check data file only once.
"""

from docopt import docopt
from glob import glob
import os
import time


TIME_GAP = 5   # 5 sec


def check_once(data_dir, lst_dir):
    data_files = glob('%s/*.lcls' % data_dir)
    curr_time = time.time()
    for data_file in data_files:
        diff_time = curr_time - os.path.getmtime(data_file)
        if diff_time > TIME_GAP:
            run_id = int(os.path.basename(data_file).split('.')[0][1:])
            lst_file = os.path.join(lst_dir, 'r%04d.lst' % run_id)
            if not os.path.exists(lst_file):
                with open(lst_file, 'w') as f:
                    f.write('%s\n' % os.path.abspath(data_file))
                print('Create new lst file %s' % lst_file)


if __name__ == '__main__':
    argv = docopt(__doc__)
    data_dir = argv['<data_dir>']
    lst_dir = argv['<lst_dir>']
    print('Monitoring %s, writing lst file to %s' % (data_dir, lst_dir))

    if argv['--only-once']:
        check_once(data_dir, lst_dir)
    else:
        while True:
            check_once(data_dir, lst_dir)
            time.sleep(5)
