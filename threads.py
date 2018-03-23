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


class CalcMeanThread(QThread):
    update_progress = pyqtSignal(float)

    def __init__(self, parent=None,
                 files=None,
                 dataset=None,
                 max_frame=0,
                 prefix=''):
        super(CalcMeanThread, self).__init__(parent)
        self.files = files
        self.dataset = dataset
        self.max_frame = max_frame
        self.prefix = prefix

    def run(self):
        count = 0
        for f in self.files:
            try:
                data = h5py.File(f, 'r')[self.dataset]
            except IOError:
                print('Failed to load %s' % f)
                continue
            if len(data.shape) == 3:
                n = data.shape[0]
                for i in range(n):
                    if count == 0:
                        img_mean = data[0].astype(np.float32)
                        count += 1
                    else:
                        img_mean += (data[i] - img_mean) / count
                        count += 1
                        ratio = count * 100. / self.max_frame
                        self.update_progress.emit(ratio)
                        if count == self.max_frame:
                            break
            else:
                if count == 0:
                    img_mean = data.value.astype(np.float32)
                    count += 1
                else:
                    img_mean += (data.value - img_mean) / count
                    count += 1
                    ratio = count * 100. / self.max_frame
                    self.update_progress.emit(ratio)
                if count >= self.max_frame:
                    break
            if count >= self.max_frame:
                break
        # write to file
        time.sleep(0.1)
        np.savez('%s.npz' % self.prefix, mean=img_mean, std=img_mean)


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
                hit_finding = 'ready'
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
            time.sleep(10)


class ConversionThread(QThread):
    progress = pyqtSignal(float)

    def __init__(self, parent=None,
                 workdir=None,
                 job=None,
                 h5_dataset=None,
                 cxi_dataset=None):
        super(ConversionThread, self).__init__(parent)
        self.workdir = workdir
        self.job = job
        self.h5_dataset = h5_dataset
        self.cxi_dataset = cxi_dataset

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
        subprocess.run(
            [shell_script,  python_script,
             h5_lst, h5_dataset, cxi_dir, cxi_lst_dir]
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
