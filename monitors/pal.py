# -*- coding: utf-8 -*-

""" Monitor raw data and generate list file at PAL-XFEL.
Usage:
    pal.py <data_dir> <lst_dir> [options]

Options:
    -h --help                   Show this screen.
    --version                   Show version.
"""

from docopt import docopt
from glob import glob
import os
import time


TIME_GAP = 5 * 60  # 5 min


if __name__ == '__main__':
    argv = docopt(__doc__)
    data_dir = argv['<data_dir>']
    lst_dir = argv['<lst_dir>']
    print('Monitoring %s, writing lst file to %s' % (data_dir, lst_dir))

    while True:
        data_files = glob('%s/*/0*.h5' % data_dir)
        curr_time = time.time()
        for data_file in data_files:
            diff_time = curr_time - os.path.getmtime(data_file)
            if diff_time > TIME_GAP:
                run_id = int(os.path.basename(data_file).split('.')[0])
                lst_file = os.path.join(lst_dir, 'r%04d.lst' % run_id)
                if not os.path.exists(lst_file):
                    with open(lst_file, 'w') as f:
                        f.write('%s\n' % data_file)
                    print('Create new lst file %s' % lst_file)
        time.sleep(5)