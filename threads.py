from PyQt5.QtCore import QThread, pyqtSignal
import h5py
import numpy as np
import time
from glob import glob
import os
import yaml
import subprocess
import math
import operator
from util import *


class CalcMeanThread(QThread):
    update_progress = pyqtSignal(float)

    def __init__(self, parent=None,
                 files=None,
                 dataset=None,
                 max_frame=0,
                 output=None):
        super(CalcMeanThread, self).__init__(parent)
        self.files = files
        self.dataset = dataset
        self.max_frame = max_frame
        self.output = output

    def run(self):
        count = 0
        for filepath in self.files:
            ext = filepath.split('.')[-1]
            if ext in ('cxi', 'h5'):
                h5_obj = h5py.File(filepath, 'r')
            else:
                h5_obj = None
            data_shape = get_data_shape(filepath)
            for i in range(data_shape[self.dataset][0]):
                img = read_image(filepath, frame=i, h5_obj=h5_obj, dataset=self.dataset).astype(np.float32)
                if count == 0:
                    img_mean = img
                else:
                    img_mean += (img - img_mean) / count
                count += 1
                ratio = count * 100. / self.max_frame
                self.update_progress.emit(ratio)
                if count == self.max_frame:
                    break
            if count >= self.max_frame:
                break
        # write to file
        time.sleep(0.1)
        if self.output is None:
            output = 'output.npz'
        else:
            output = self.output
        dir_ = os.path.dirname((output))
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        np.savez(output, mean=img_mean, std=img_mean)


class CrawlerThread(QThread):
    jobs = pyqtSignal(list)
    conf = pyqtSignal(list)

    def __init__(self, parent=None, workdir=None):
        super(CrawlerThread, self).__init__(parent)
        self.workdir = workdir

    def run(self):
        while True:
            # check data from h5 lst
            job_list = []
            lst_files = glob('%s/h5_lst/*.lst' % self.workdir)
            for lst_file in lst_files:
                time1 = os.path.getmtime(lst_file)
                job_name = os.path.basename(lst_file).split('.')[0]
                # check h52cxi status
                cxi_raw = os.path.join(self.workdir, 'cxi_raw', job_name)
                h52cxi = 'ready'
                raw_frames = 0
                hits = 0
                hit_rate = 0
                comp_ratio = 0
                if os.path.isdir(cxi_raw):
                    progress_file = os.path.join(cxi_raw, 'progress.txt')
                    if os.path.exists(progress_file):
                        with open(progress_file, 'r') as f:
                            h52cxi = f.readline()
                    stat_file = os.path.join(cxi_raw, 'stat.yml')
                    if os.path.exists(stat_file):
                        with open(stat_file, 'r') as f:
                            stat = yaml.load(f)
                            raw_frames = stat['total frames']
                            comp_ratio = stat['compression ratio']
                # check cxi lst status
                cxi_lst = os.path.join(self.workdir, 'cxi_lst', '%s.lst' % job_name)
                if os.path.exists(cxi_lst):
                    hit_finding = 'ready'
                else:
                    hit_finding = 'not ready'
                # check hit finding status
                cxi_hit_dir = os.path.join(self.workdir, 'cxi_hit', job_name)
                hit_tags = glob('%s/*' % cxi_hit_dir)
                tags = [os.path.basename(tag) for tag in hit_tags]
                if len(hit_tags) == 0:
                    tag = ''
                    time2 = math.inf
                    job_list.append(
                        {
                            'job': job_name,
                            'tag': tag,
                            'h52cxi': h52cxi,
                            'hit finding': hit_finding,
                            'raw frames': raw_frames,
                            'hits': hits,
                            'hit rate': '%.2f%%' % hit_rate,
                            'compression ratio': '%.2f' % comp_ratio,
                            'time1': time1,
                            'time2': time2,
                        }
                    )
                else:
                    for tag in tags:
                        tag_dir = os.path.join(
                            self.workdir, 'cxi_hit', job_name, tag
                        )
                        stat_file = os.path.join(tag_dir, 'stat.yml')
                        if os.path.exists(stat_file):
                            with open(stat_file, 'r') as f:
                                stat = yaml.load(f)
                            time2 = os.path.getmtime(stat_file)
                            progress_file = os.path.join(
                                tag_dir, 'progress.txt'
                            )
                            if os.path.exists(progress_file):
                                with open(progress_file, 'r') as f:
                                    hit_finding = f.readline()
                            hits = stat['total hits']
                            hit_rate = stat['hit rate']
                        else:
                            time2 = math.inf
                        job_list.append(
                            {
                                'job': job_name,
                                'tag': tag,
                                'h52cxi': h52cxi,
                                'hit finding': hit_finding,
                                'raw frames': raw_frames,
                                'hits': hits,
                                'hit rate': '%.2f%%' % hit_rate,
                                'compression ratio': '%.2f' % comp_ratio,
                                'time1': time1,
                                'time2': time2,
                            }
                        )
                job_list = sorted(
                    job_list, key=operator.itemgetter('time1', 'time2')
                )
            self.jobs.emit(job_list)

            # check hit finding conf files
            conf_dir = os.path.join(self.workdir, 'conf.d')
            conf_files = glob('%s/*.yml' % conf_dir)
            self.conf.emit(conf_files)
            time.sleep(5)


class ConversionThread(QThread):
    progress = pyqtSignal(float)

    def __init__(self, parent=None,
                 workdir=None,
                 job=None,
                 h5_dataset=None,
                 cxi_dataset=None,
                 cxi_size=1000,
                 cxi_dtype='int32'):
        super(ConversionThread, self).__init__(parent)
        self.workdir = workdir
        self.job = job
        self.h5_dataset = h5_dataset
        self.cxi_dataset = cxi_dataset
        self.cxi_size = cxi_size
        self.cxi_dtype = cxi_dtype

    def run(self):
        workdir = self.workdir
        job = self.job
        h5_lst = os.path.join(workdir, 'h5_lst', '%s.lst' % job)
        h5_dataset = self.h5_dataset
        cxi_dataset = self.cxi_dataset
        cxi_dir = os.path.join(workdir, 'cxi_raw', job)
        cxi_lst_dir = os.path.join(workdir, 'cxi_lst')
        shell_script = './scripts/run_h52cxi_local'
        python_script = './batch_h52cxi.py'
        cxi_size = str(self.cxi_size)
        cxi_dtype = str(self.cxi_dtype)
        subprocess.run(
            [
                shell_script,  python_script,
                h5_lst, h5_dataset, cxi_dir, cxi_lst_dir,
                '--cxi-dataset', cxi_dataset,
                '--cxi-size', cxi_size,
                '--cxi-dtype', cxi_dtype,
             ]
        )


class HitFindingThread(QThread):
    def __init__(self, parent=None,
                 workdir=None,
                 job=None,
                 conf=None,
                 tag=None):
        super(HitFindingThread, self).__init__(parent)
        self.workdir = workdir
        self.job = job
        self.conf = conf
        self.tag = tag

    def run(self):
        cxi_lst = os.path.join(self.workdir, 'cxi_lst', '%s.lst' % self.job)
        conf = self.conf
        hit_dir = os.path.join(self.workdir, 'cxi_hit', self.job, self.tag)
        shell_script = './scripts/run_hit_finding_local'
        python_script = './batch_hit_finding.py'
        subprocess.run([shell_script, python_script, cxi_lst, conf, hit_dir])
