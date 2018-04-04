from PyQt5.QtCore import QThread, pyqtSignal

import os
import subprocess
import operator
import time
from glob import glob
import yaml

import numpy as np
import math
import random
import h5py
from util import get_data_shape, read_image


class MeanCalculatorThread(QThread):
    update_progress = pyqtSignal(float)

    def __init__(self, files, dataset, max_frame=0, output=None):
        super(MeanCalculatorThread, self).__init__()
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
                img = read_image(
                    filepath, frame=i, h5_obj=h5_obj,
                    dataset=self.dataset).astype(np.float32)
                if count == 0:
                    img_mean = img
                    img_sigma = np.zeros_like(img_mean)
                else:
                    img_mean_prev = img_mean.copy()
                    img_mean += (img - img_mean) / (count + 1.)
                    img_sigma += (img - img_mean_prev) * (img - img_mean)
                count += 1
                ratio = count * 100. / self.max_frame
                self.update_progress.emit(ratio)
                if count == self.max_frame:
                    break
            if count >= self.max_frame:
                break
        img_sigma = img_sigma / count
        img_sigma = np.sqrt(img_sigma)
        # write to file
        time.sleep(0.1)
        if self.output is None:
            output = 'output.npz'
        else:
            output = self.output
        dir_ = os.path.dirname(output)
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        np.savez(output, mean=img_mean, sigma=img_sigma)


class GenPowderThread(QThread):
    info = pyqtSignal(str)

    def __init__(self, files, conf_file, settings,
                 max_frame=0,
                 output_dir=None,
                 prefix='powder'):
        super(GenPowderThread, self).__init__()
        self.files = files
        self.conf_file = conf_file
        self.max_frame = max_frame
        self.output_dir = output_dir
        self.prefix = prefix
        self.settings = settings

    def run(self):
        # make file lst
        file_lst = '.powder-%d.lst' % random.randint(0, 99999)
        with open(file_lst, 'w') as f:
            for i in range(len(self.files)):
                f.write('%s\n' % self.files[i])
        conf_file = self.conf_file
        if self.output_dir is None:
            outout_dir = 'output'
        else:
            output_dir = self.output_dir
        prefix = self.prefix
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/scripts/run_powder_generator_%s' % (
            dir_, self.settings.script_suffix)
        python_script = '%s/mpi/batch_peak_powder.py' % dir_

        self.info.emit('Submitting powder generation task.')
        subprocess.run(
            [
                shell_script, python_script,
                file_lst, conf_file,
                '-o', output_dir,
                '-p', prefix,
            ]
        )
        self.info.emit('Powder generation done!')


class CrawlerThread(QThread):
    jobs = pyqtSignal(list)
    conf = pyqtSignal(list)
    stat = pyqtSignal(dict)

    def __init__(self, parent=None, workdir=None):
        super(CrawlerThread, self).__init__(parent)
        self.workdir = workdir

    def run(self):
        while True:
            # check data from h5 lst
            job_list = []
            raw_lst_files = glob('%s/raw_lst/*.lst' % self.workdir)
            total_raw_frames = 0
            total_processed_frames = {}
            total_processed_hits = {}
            for raw_lst in raw_lst_files:
                time1 = os.path.getmtime(raw_lst)
                job_name = os.path.basename(raw_lst).split('.')[0]
                # check compression status
                cxi_comp_dir = os.path.join(self.workdir, 'cxi_comp', job_name)
                compression = 'ready'
                hit_rate = 0
                comp_ratio = 0
                raw_frames = 0
                if os.path.isdir(cxi_comp_dir):
                    stat_file = os.path.join(cxi_comp_dir, 'stat.yml')
                    if os.path.exists(stat_file):
                        with open(stat_file, 'r') as f:
                            stat = yaml.load(f)
                            if stat is None:
                                stat = {}
                            compression = stat.get('progress', '0')
                            raw_frames = stat.get('total frames', 0)
                            comp_ratio = stat.get('compression ratio', 0)
                            if raw_frames is not None:
                                total_raw_frames += raw_frames
                # check cxi lst status
                cxi_lst = os.path.join(
                    self.workdir, 'cxi_lst', '%s.lst' % job_name)
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
                    processed_frames = 0
                    processed_hits = 0
                    csv2cxi = 0
                    job_list.append(
                        {
                            'job': job_name,
                            'raw frames': raw_frames,
                            'compression': compression,
                            'compression ratio': comp_ratio,
                            'tag': tag,
                            'hit finding': hit_finding,
                            'processed frames': processed_frames,
                            'processed hits': processed_hits,
                            'hit rate': hit_rate,
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
                                if stat is None:
                                    stat = {}
                            time2 = stat.get('time start', math.inf)
                            processed_hits = stat.get('processed hits', 0)
                            processed_frames = stat.get('processed frames', 0)
                            hit_finding = stat.get('progress', 0)
                            if tag in total_processed_hits.keys():
                                total_processed_hits[tag] += processed_hits
                            else:
                                total_processed_hits[tag] = processed_hits
                            if tag in total_processed_frames.keys():
                                total_processed_frames[tag] += processed_frames
                            else:
                                total_processed_frames[tag] = processed_frames
                            hit_rate = stat.get('hit rate')
                        else:
                            time2 = math.inf
                            processed_frames = 0
                            processed_hits = 0
                        job_list.append(
                            {
                                'job': job_name,
                                'raw frames': raw_frames,
                                'compression': compression,
                                'compression ratio': comp_ratio,
                                'tag': tag,
                                'hit finding': hit_finding,
                                'processed hits': processed_hits,
                                'processed frames': processed_frames,
                                'hit rate': hit_rate,
                                'time1': time1,
                                'time2': time2,
                            }
                        )
                job_list = sorted(
                    job_list, key=operator.itemgetter('job', 'time2')
                )
            self.jobs.emit(job_list)

            # check hit finding conf files
            conf_dir = os.path.join(self.workdir, 'conf')
            conf_files = glob('%s/*.yml' % conf_dir)
            self.conf.emit(conf_files)

            # check stat
            stat = {
                'total raw frames': total_raw_frames,
                'total processed frames': total_processed_frames,
                'total processed hits': total_processed_hits,
            }
            self.stat.emit(stat)
            time.sleep(5)


class CompressorThread(QThread):
    progress = pyqtSignal(float)

    def __init__(self, job, settings, parent=None):
        super(CompressorThread, self).__init__(parent)
        self.job = job
        self.settings = settings

    def run(self):
        job = self.job
        workdir = self.settings.workdir
        raw_lst = os.path.join(workdir, 'raw_lst', '%s.lst' % job)
        raw_dataset = self.settings.raw_dataset
        comp_dataset = self.settings.comp_dataset
        comp_dir = os.path.join(workdir, 'cxi_comp', job)
        comp_lst_dir = os.path.join(workdir, 'cxi_lst')
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/scripts/run_compressor_%s' \
                       % (dir_, self.settings.script_suffix)
        python_script = '%s/mpi/batch_compressor.py' % dir_
        comp_size = str(self.settings.comp_size)
        comp_dtype = str(self.settings.comp_dtype)
        print(shell_script, job, python_script)
        subprocess.run(
            [
                shell_script,  job, python_script,
                raw_lst, raw_dataset, comp_dir, comp_lst_dir,
                '--comp-dataset', comp_dataset,
                '--comp-size', comp_size,
                '--comp-dtype', comp_dtype,
             ]
        )


class HitFinderThread(QThread):
    def __init__(self, parent=None,
                 workdir=None,
                 job=None,
                 conf=None,
                 tag=None):
        super(HitFinderThread, self).__init__(parent)
        self.workdir = workdir
        self.job = job
        self.conf = conf
        self.tag = tag

    def run(self):
        cxi_lst = os.path.join(self.workdir, 'cxi_lst', '%s.lst' % self.job)
        conf = self.conf
        hit_dir = os.path.join(self.workdir, 'cxi_hit', self.job, self.tag)
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/scripts/run_hit_finder_PAL7' % dir_
        python_script = '%s/mpi/batch_hit_finder.py' % dir_
        subprocess.run([shell_script, python_script, cxi_lst, conf, hit_dir])
