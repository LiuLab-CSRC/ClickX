from PyQt5.QtCore import QThread, pyqtSignal
import h5py
import numpy as np
import time
from glob import glob
import os


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

    def __init__(self, parent=None, workdir=None):
        super(CrawlerThread, self).__init__(parent)
        self.workdir = workdir

    def run(self):
        while True:
            job_list = []
            lst_files = glob('%s/h5_lst/*.lst' % self.workdir)
            for lst_file in lst_files:
                job_name = os.path.basename(lst_file).split('.')[0]
                job_tag = ''
                job_id = '%s-%s' % (job_name, job_tag)
                job_list.append(
                    {
                        'name': job_name,
                        'id': job_id,
                        'path': lst_file,
                        'tag': job_tag,
                    }
                )
            self.jobs.emit(job_list)
            time.sleep(5)


class ConversionThread(QThread):
    progress = pyqtSignal(float)

    def __init__(self, parent=None, workdir=None):
        pass


class HitFindingThread(QThread):
    def __init__(self, parent=None, workdir=None):
        super(HitFindingThread, self).__init__()
        pass        
