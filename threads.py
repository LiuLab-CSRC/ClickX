# -*- coding: utf-8 -*-


from PyQt5.QtCore import QThread, pyqtSignal

import os
import subprocess
import time

import numpy as np
import random
import h5py
from util import util


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
            h5_obj = h5py.File(filepath, 'r') if ext in ('cxi', 'h5') else None
            lcls_data = util.get_lcls_data(filepath) if ext == 'lcls' else None
            data_shape = util.get_data_shape(filepath)
            for i in range(data_shape[self.dataset][0]):
                img = util.read_image(
                    filepath, frame=i,
                    h5_obj=h5_obj,
                    lcls_data=lcls_data,
                    dataset=self.dataset
                )['image']
                if img is None:
                    continue
                img = img.astype(np.float64)
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
                 output=None):
        super(GenPowderThread, self).__init__()
        self.files = files
        self.conf_file = conf_file
        self.max_frame = max_frame
        self.output = output
        self.settings = settings

    def run(self):
        # make file lst
        file_lst = '.powder-%d.lst' % random.randint(0, 99999)
        with open(file_lst, 'w') as f:
            for i in range(len(self.files)):
                f.write('%s\n' % self.files[i])
        conf_file = self.conf_file
        if self.output is None:
            output = 'output'
        else:
            output = self.output
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/engines/%s/run_powder_generator' % (
            dir_, self.settings.engine)
        python_script = '%s/util/batch_peak_powder.py' % dir_

        subprocess.call(
            [
                shell_script, python_script,
                file_lst, conf_file,
                '-o', output,
                '--max-frames', str(self.max_frame),
                '--flush'
            ]
        )


class CompressorThread(QThread):
    def __init__(self, job, settings, parent=None):
        super(CompressorThread, self).__init__(parent)
        self.job = job
        self.settings = settings

    def run(self):
        job = self.job
        raw_lst = os.path.join('raw_lst', '%s.lst' % job)
        raw_dataset = self.settings.raw_dataset
        comp_dataset = self.settings.compressed_dataset
        comp_dir = os.path.join('cxi_comp', job)
        comp_lst_dir = 'cxi_lst'
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/engines/%s/run_compressor' \
                       % (dir_, self.settings.engine)
        python_script = '%s/util/batch_compressor.py' % dir_
        comp_size = str(self.settings.compressed_batch_size)
        subprocess.call(
            [
                shell_script,  job, python_script,
                raw_lst, raw_dataset, comp_dir, comp_lst_dir,
                '--comp-dataset', comp_dataset,
                '--comp-size', comp_size,
            ]
        )


class HitFinderThread(QThread):
    def __init__(self, settings,
                 parent=None,
                 job=None,
                 tag=None,
                 compressed=True):
        super(HitFinderThread, self).__init__(parent)
        self.settings = settings
        self.job = job
        self.tag = tag
        self.compressed = compressed

    def run(self):
        if self.compressed:
            cxi_lst = os.path.join('cxi_lst', '%s.lst' % self.job)
        else:
            cxi_lst = os.path.join('raw_lst', '%s.lst' % self.job)
        conf = 'conf/hit_finding/%s.yml' % self.tag
        job = self.job
        min_peaks = str(self.settings.min_peaks)
        min_max_intensity = str(self.settings.min_max_intensity)
        hit_dir = os.path.join('cxi_hit', self.job, self.tag)
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/engines/%s/run_hit_finder' % \
                       (dir_, self.settings.engine)
        python_script = '%s/util/batch_hit_finder.py' % dir_
        subprocess.call(
            [
                shell_script, job, python_script, cxi_lst, conf, hit_dir,
                '--min-peaks', min_peaks,
                '--min-max-intensity', min_max_intensity
            ]
        )


class Peak2CxiThread(QThread):
    def __init__(self, settings, job, tag):
        super(Peak2CxiThread, self).__init__()
        self.settings = settings
        self.job = job
        self.tag = tag

    def run(self):
        hit_dir = os.path.join('cxi_hit', self.job, self.tag)
        options = []
        mask_file = os.path.join(hit_dir, 'mask.npy')
        if os.path.exists(mask_file):
            options += ['--mask-file', mask_file]
        peak_file = os.path.join(hit_dir, '%s.npy' % self.job)
        min_peaks = str(self.settings.min_peaks)
        raw_data_path = self.settings.cxi_raw_data_path
        peak_info_path = self.settings.cxi_peak_info_path
        extra_datasets = self.settings.cheetah_datasets
        if len(extra_datasets) > 0:
            options += ['--extra-datasets', extra_datasets]
        batch_size = str(self.settings.mpi_batch_size)
        cxi_size = str(self.settings.cxi_size)
        options += ['--min-peaks', min_peaks,
                    '--batch-size', batch_size,
                    '--raw-data-path', raw_data_path,
                    '--peak-info-path', peak_info_path,
                    '--cxi-size', cxi_size]
        dir_ = os.path.dirname(__file__)
        shell_script = '%s/engines/%s/run_peak2cxi' % \
                       (dir_, self.settings.engine)
        python_script = '%s/util/batch_peak2cxi.py' % dir_
        subprocess.call(
            [shell_script, self.job, python_script, peak_file, hit_dir]
            + options
        )
