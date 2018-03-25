from PyQt5.QtCore import QThread, pyqtSignal
import time
from glob import glob
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
        dir_ = os.path.dirname((output))
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        np.savez(output, mean=img_mean, sigma=img_sigma)


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
            lst_files = glob('%s/h5_lst/*.lst' % self.workdir)
            total_raw_frames = 0
            total_processed_frames = {}
            total_processed_hits = {}
            for lst_file in lst_files:
                time1 = os.path.getmtime(lst_file)
                job_name = os.path.basename(lst_file).split('.')[0]
                # check h52cxi status
                cxi_raw = os.path.join(self.workdir, 'cxi_raw', job_name)
                h52cxi = 'ready'
                hit_rate = 0
                comp_ratio = 0
                raw_frames = 0
                if os.path.isdir(cxi_raw):
                    stat_file = os.path.join(cxi_raw, 'stat.yml')
                    if os.path.exists(stat_file):
                        with open(stat_file, 'r') as f:
                            stat = yaml.load(f)
                            if stat is None:
                                stat = {}
                            h52cxi = stat.get('progress', '0')
                            raw_frames = stat.get('total frames', 0)
                            comp_ratio = stat.get('compression ratio', 0)
                            if raw_frames is not None:
                                total_raw_frames += raw_frames
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
                    processed_frames = 0
                    processed_hits = 0
                    job_list.append(
                        {
                            'job': job_name,
                            'tag': tag,
                            'h52cxi': h52cxi,
                            'hit finding': hit_finding,
                            'raw frames': raw_frames,
                            'processed frames': processed_frames,
                            'processed hits': processed_hits,
                            'hit rate': hit_rate,
                            'compression ratio': comp_ratio,
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
                                'tag': tag,
                                'h52cxi': h52cxi,
                                'hit finding': hit_finding,
                                'raw frames': raw_frames,
                                'processed hits': processed_hits,
                                'processed frames': processed_frames,
                                'hit rate': hit_rate,
                                'compression ratio': comp_ratio,
                                'time1': time1,
                                'time2': time2,
                            }
                        )
                job_list = sorted(
                    job_list, key=operator.itemgetter('time1', 'time2')
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
                 tag=None,
                 cxi_size=100):
        super(HitFindingThread, self).__init__(parent)
        self.workdir = workdir
        self.job = job
        self.conf = conf
        self.tag = tag
        self.cxi_size = cxi_size

    def run(self):
        cxi_lst = os.path.join(self.workdir, 'cxi_lst', '%s.lst' % self.job)
        conf = self.conf
        hit_dir = os.path.join(self.workdir, 'cxi_hit', self.job, self.tag)
        shell_script = './scripts/run_hit_finding_local'
        python_script = './batch_hit_finding.py'
        cxi_size = str(self.cxi_size)
        subprocess.run([shell_script, python_script, cxi_lst, conf, hit_dir, '--cxi-size', cxi_size])
