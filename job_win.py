from os.path import abspath, dirname

from PyQt5.QtCore import pyqtSlot, QPoint, Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QMenu
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.uic import loadUi

from threads import *


class JobWindow(QWidget):
    def __init__(self, parent=None,
                 workdir=None,
                 h5_dataset=None,
                 cxi_dataset=None,
                 cxi_size=None,
                 cxi_dtype=None,
                 header_labels=None):
        super(JobWindow, self).__init__(parent)
        # setup ui
        loadUi('ui/jobs.ui', self)
        if workdir is None:
            self.workdir = dirname(abspath(__file__))
        else:
            self.workdir = workdir
        self.h5_dataset = h5_dataset
        self.cxi_dataset = cxi_dataset
        self.workdir_lineedit.setText(self.workdir)
        self.h5_dataset_lineedit.setText(self.h5_dataset)
        self.cxi_dataset_lineedit.setText(self.cxi_dataset)
        self.header_labels = header_labels
        self.job_table.setHorizontalHeaderLabels(self.header_labels)
        self.cxi_size = cxi_size
        self.cxi_dtype = cxi_dtype

        self.curr_conf = []
        self.crawler_running = False

        # threads
        self.crawler_thread = None
        self.conv_thread = None
        self.hit_finding_thread = None

        # slots
        self.browse_btn.clicked.connect(self.select_workdir)
        self.crawler_btn.clicked.connect(self.start_or_stop_crawler)
        self.job_table.customContextMenuRequested.connect(self.show_job_menu)

    @pyqtSlot()
    def select_workdir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Select a directory', self.workdir)
        if len(directory) > 0:
            self.workdir = directory
            self.workdir_lineedit.setText(self.workdir)

    @pyqtSlot()
    def start_or_stop_crawler(self):
        if self.crawler_running:
            self.crawler_running = False
            self.crawler_btn.setText('start crawler')
            self.crawler_thread.terminate()
        else:
            self.crawler_running = True
            self.crawler_btn.setText('stop crawler')
            self.crawler_thread = CrawlerThread(workdir=self.workdir)
            self.crawler_thread.jobs.connect(self.update_jobs)
            self.crawler_thread.conf.connect(self.update_conf)
            self.crawler_thread.stat.connect(self.update_stat)
            self.crawler_thread.start()

    @pyqtSlot(list)
    def update_jobs(self, jobs):
        for i in range(len(jobs)):
            job = jobs[i]
            self.fill_table_row(job, i)

    @pyqtSlot(list)
    def update_conf(self, confs):
        for conf in confs:
            if conf in self.curr_conf:
                continue
            tag = os.path.basename(conf).split('.')[0]
            self.hit_finding_conf.addItem(tag, userData=conf)
            self.curr_conf.append(conf)

    @pyqtSlot(dict)
    def update_stat(self, stat):
        total_raw_frames = stat.get('total raw frames', 0)
        self.total_raw_frame_label.setText(str(total_raw_frames))
        curr_id = self.hit_finding_conf.currentIndex()
        tag = self.hit_finding_conf.itemText(curr_id)
        total_processed_frame_dict = stat.get('total processed frames', {})
        total_processed_hits_dict = stat.get('total processed hits', {})
        if tag in stat['total processed frames'].keys():
            total_processed_hits = total_processed_hits_dict.get(tag, 0)
            total_processed_frames = total_processed_frame_dict.get(tag, 0)
            if total_processed_frames != 0:
                total_hit_rate = float(total_processed_hits) / total_processed_frames * 100.
            else:
                total_hit_rate = 0.
            self.processed_hit_label.setText(str(total_processed_hits))
            self.processed_frame_label.setText(str(total_processed_frames))
            self.hit_rate_label.setText('%.2f%%' % total_hit_rate)

    @pyqtSlot(QPoint)
    def show_job_menu(self, pos):
        job_table = self.job_table
        menu = QMenu()
        row = job_table.currentRow()
        job = job_table.item(row, 0).text()
        workdir = self.workdir_lineedit.text()
        action_h52cxi = menu.addAction('run h52cxi')
        action_hit_finding = menu.addAction('run hit finding')
        action = menu.exec_(job_table.mapToGlobal(pos))
        if action == action_h52cxi:
            self.conv_thread = ConversionThread(
                workdir=workdir,
                job=job,
                h5_dataset=self.h5_dataset_lineedit.text(),
                cxi_dataset=self.cxi_dataset_lineedit.text(),
                cxi_size=self.cxi_size,
                cxi_dtype=self.cxi_dtype,
            )
            self.conv_thread.start()
        elif action == action_hit_finding:
            curr_id = self.hit_finding_conf.currentIndex()
            conf = self.hit_finding_conf.itemData(curr_id)
            tag = self.hit_finding_conf.itemText(curr_id)
            self.hit_finding_thread = HitFindingThread(
                workdir=workdir,
                job=job,
                conf=conf,
                tag=tag,
                cxi_size=self.cxi_size,
            )
            self.hit_finding_thread.start()

    def fill_table_row(self, row_dict, row):
        row_count = self.job_table.rowCount()
        if row_count == row:
            self.job_table.insertRow(row_count)
        for col, field in enumerate(self.header_labels):
            item = self.job_table.item(row, col)
            if item is None:
                item = QTableWidgetItem(row_dict[field])
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.job_table.setItem(row, col, item)
            else:
                item.setText(str(row_dict[field]))
