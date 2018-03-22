from PyQt5.QtCore import QThread, pyqtSignal
import h5py
import numpy as np


class CalcMeanThread(QThread):
    update_progress = pyqtSignal(float)

    def __init__(self, parent=None, files=None, dataset=None, max_frame=0):
        super(CalcMeanThread, self).__init__(parent)
        self.files = files
        self.dataset = dataset
        self.max_frame = max_frame

    def run(self):
        count = 0
        for f in self.files:
            try:
                data = h5py.File(f, 'r')[self.dataset]
            except IOError:
                continue
            if len(data.shape) == 3:
                n = data.shape[0]
                if count == 0:
                    img_mean = data[0].astype(np.float32)
                    count += 1
                else:
                    for i in range(n):
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
