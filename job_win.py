# -*- coding: utf-8 -*-


import csv
import os
from glob import glob
import yaml
import time
import operator
import subprocess

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QPoint, Qt, QTimer
from PyQt5.QtWidgets import QWidget, QMenu, QTableWidgetItem
from PyQt5.uic import loadUi
from threads import CompressorThread, HitFinderThread, Peak2CxiThread


class JobWindow(QWidget):
    view_hits = pyqtSignal(str, str)

    def __init__(self, settings, main_win):
        super(JobWindow, self).__init__()
        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/window/batch.ui' % dir_, self)
        self.settings = settings
        self.main_win = main_win
        self.raw_dataset = settings.raw_dataset
        self.compressed_dataset = settings.compressed_dataset
        self.hit_tag = settings.curr_hit_tag

        if settings.compress_raw_data:
            self.header_labels = (
                'job id', 'compression', 'compression ratio',
                'raw frames', 'tag id', 'hit finding',
                'processed frames', 'processed hits', 'hit rate',
                'peak2cxi'
            )
        else:
            self.header_labels = (
                'job id', 'raw frames', 'tag id', 'hit finding',
                'processed frames', 'processed hits', 'hit rate',
                'peak2cxi'
            )
        self.jobTable.setColumnCount(len(self.header_labels))
        self.jobTable.setHorizontalHeaderLabels(self.header_labels)
        self.mpi_batch_size = settings.mpi_batch_size

        self.jobs_info = None
        self.conf_files = None
        self.jobs_stat = None
        self.auto_submit = False
        self.jobs = []

        self.timer = QTimer()

        # slots
        self.jobTable.customContextMenuRequested.connect(self.show_job_menu)
        self.autoSubmit.toggled.connect(self.change_auto_submit)
        self.exportButton.clicked.connect(self.export_jobs)
        self.timer.timeout.connect(self.check_and_submit_jobs)

    def start(self):
        self.timer.start(self.settings.update_period * 1000)

    def update_settings(self, settings):
        self.settings = settings
        self.raw_dataset = settings.raw_dataset
        self.compressed_dataset = settings.compressed_dataset
        self.hit_tag = settings.curr_hit_tag
        self.mpi_batch_size = settings.mpi_batch_size

        if settings.compress_raw_data:
            self.header_labels = (
                'job id', 'compression', 'compression ratio',
                'raw frames', 'tag id', 'hit finding',
                'processed frames', 'processed hits', 'hit rate',
                'peak2cxi'
            )
        else:
            self.header_labels = (
                'job id', 'raw frames', 'tag id', 'hit finding',
                'processed frames', 'processed hits', 'hit rate',
                'peak2cxi'
            )

        self.jobTable.setColumnCount(len(self.header_labels))
        self.jobTable.setHorizontalHeaderLabels(self.header_labels)

    def crawler_run(self):
        jobs_info = []
        total_raw_frames = 0
        total_processed_frames = 0
        total_processed_hits = 0
        raw_lst_files = glob('raw_lst/*.lst')
        for raw_lst in raw_lst_files:
            job_id = os.path.basename(raw_lst).split('.')[0]
            if self.settings.compress_raw_data:
                cxi_comp_dir = os.path.join('cxi_comp', job_id)
                cxi_comp_stat = check_cxi_comp(cxi_comp_dir)
                compression = cxi_comp_stat.get('progress', 0)
                compression_ratio = cxi_comp_stat.get('compression ratio', 0)
                raw_frames = cxi_comp_stat.get('total frames', 0)

                cxi_lst = os.path.join('cxi_lst', '%s.lst' % job_id)
                if os.path.exists(cxi_lst):
                    hit_finding = 'ready'
                else:
                    hit_finding = 'not ready'
            else:
                hit_finding = 'ready'
                compression = 'NA'
                compression_ratio = 'NA'
                raw_frames = 0
            cxi_hit_dir = os.path.join('cxi_hit', job_id)
            hit_tags = glob('%s/*' % cxi_hit_dir)
            if len(hit_tags) == 0:
                tag_id = 'NA'
                jobs_info.append(
                    {
                        'job id': job_id,
                        'compression': compression,
                        'compression ratio': compression_ratio,
                        'raw frames': raw_frames,
                        'tag id': tag_id,
                        'hit finding': hit_finding,
                        'processed frames': 0,
                        'processed hits': 0,
                        'hit rate': 0,
                        'peak2cxi': 'not ready'
                    }
                )
                total_raw_frames += raw_frames
            else:
                tag_ids = [os.path.basename(tag_id) for tag_id in hit_tags]
                for tag_id in tag_ids:
                    tag_dir = os.path.join('cxi_hit', job_id, tag_id)
                    cxi_hit_stat = check_cxi_hit(tag_dir)
                    hit_finding = cxi_hit_stat.get('progress', 0)
                    processed_frames = cxi_hit_stat.get('processed frames', 0)
                    processed_hits = cxi_hit_stat.get('processed hits', 0)
                    hit_rate = cxi_hit_stat.get('hit rate', 0)
                    raw_frames = cxi_hit_stat.get('total frames', 0)
                    peak2cxi_file = os.path.join(tag_dir, 'peak2cxi.yml')
                    if hit_finding == 'done':
                        if not os.path.exists(peak2cxi_file):
                            peak2cxi = 'ready'
                        else:
                            peak2cxi_stat = check_peak2cxi(tag_dir)
                            peak2cxi = peak2cxi_stat.get('progress', 0)
                    else:
                        peak2cxi = 'not ready'
                    jobs_info.append(
                        {
                            'job id': job_id,
                            'compression': compression,
                            'compression ratio': compression_ratio,
                            'raw frames': raw_frames,
                            'tag id': tag_id,
                            'hit finding': hit_finding,
                            'processed frames': processed_frames,
                            'processed hits': processed_hits,
                            'hit rate': hit_rate,
                            'peak2cxi': peak2cxi,
                        }
                    )
                    if tag_id == self.hit_tag:
                        total_raw_frames += raw_frames
                        total_processed_frames += processed_frames
                        total_processed_hits += processed_hits

        self.jobs_info = sorted(
            jobs_info, key=operator.itemgetter('job id')
        )

        if total_processed_frames > 0:
            total_hit_rate = float(total_processed_hits) / \
                             total_processed_frames
        else:
            total_hit_rate = 0
        self.jobs_stat = {
            'total raw frames': total_raw_frames,
            'total processed frames': total_processed_frames,
            'total processed hits': total_processed_hits,
            'total hit rate': total_hit_rate,
        }

    @pyqtSlot(list)
    def update_jobs_info(self):
        for i, job_info in enumerate(self.jobs_info):
            self.fill_table_row(job_info, i)

    @pyqtSlot(dict)
    def update_stat(self):
        self.rawFrames.setText(
            str(self.jobs_stat.get('total raw frames', 0)))
        self.processedHits.setText(
            str(self.jobs_stat.get('total processed hits', 0)))
        self.processedFrames.setText(
            str(self.jobs_stat.get('total processed frames', 0)))
        self.hitRate.setText(
            '%.2f%%' % (self.jobs_stat.get('total hit rate', 0) * 100))

    @pyqtSlot(QPoint)
    def show_job_menu(self, pos):
        menu = QMenu()
        items = self.jobTable.selectedItems()
        rows = []
        for item in items:
            rows.append(self.jobTable.row(item))
        rows = set(rows)  # remove duplicates
        if self.settings.compress_raw_data:
            action_compression = menu.addAction('run compressor')
        action_hit_finding = menu.addAction('run hit finder')
        action_peak2cxi = menu.addAction('save all to cxi')
        menu.addSeparator()
        action_view_hits = menu.addAction('view hits')
        menu.addSeparator()
        action_sum = menu.addAction('calc sum')
        action = menu.exec_(self.jobTable.mapToGlobal(pos))

        if self.settings.compress_raw_data and action == action_compression:
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')).text()
                job = Job(job_type='compression',
                          settings=self.settings,
                          job_id=job_id,
                          tag_id='NA')
                self.jobs.append(job)
                job.submit()
        elif action == action_hit_finding:
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')
                ).text()
                job = Job(job_type='hit finding',
                          settings=self.settings,
                          job_id=job_id,
                          tag_id=self.hit_tag,
                          compressed=self.settings.compress_raw_data)
                self.jobs.append(job)
                job.submit()
        elif action == action_peak2cxi:
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')).text()
                tag_id = self.jobTable.item(
                    row, self.header_labels.index('tag id')).text()
                job = Job(job_type='peak2cxi',
                          settings=self.settings,
                          job_id=job_id,
                          tag_id=tag_id)
                self.jobs.append(job)
                job.submit()
        elif action == action_view_hits:
            row = self.jobTable.currentRow()
            job = self.jobTable.item(
                row, self.header_labels.index('job id')).text()
            tag = self.jobTable.item(
                row, self.header_labels.index('tag id')).text()
            self.view_hits.emit(job, tag)
        elif action == action_sum:
            s = 0
            for item in items:
                try:
                    s += int(item.text())
                except ValueError:
                    print('%s not a number' % item.text())
            print('sum of selected items: %.2f' % s)

    def fill_table_row(self, row_dict, row):
        row_count = self.jobTable.rowCount()
        if row_count == row:
            self.jobTable.insertRow(row_count)
        for col, field in enumerate(self.header_labels):
            item = self.jobTable.item(row, col)
            if item is None:
                item = QTableWidgetItem(row_dict[field])
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.jobTable.setItem(row, col, item)
            else:
                item.setText(str(row_dict[field]))

    @pyqtSlot(bool)
    def change_auto_submit(self, auto_submit):
        self.auto_submit = auto_submit

    @pyqtSlot()
    def check_and_submit_jobs(self):
        print('checking and submitting jobs')
        self.crawler_run()
        self.update_jobs_info()
        self.update_stat()
        if self.auto_submit:
            self.find_and_submit_jobs()
        if len(self.settings.raw_data_dir) > 0:
            if self.settings.facility == 'PAL':
                dir_ = os.path.dirname(__file__)
                monitor_script = os.path.join(dir_, 'monitors', 'pal.py')
                raw_data_dir = self.settings.raw_data_dir
                raw_lst_dir = 'raw_lst'
                subprocess.call(
                    [monitor_script, raw_data_dir, raw_lst_dir, '--only-once']
                )
        else:
            print('pal-monitor failed to start, please set raw data directory in settings')

    def find_and_submit_jobs(self):
        nb_total_jobs = 0
        nb_ready_jobs = 0
        nb_not_ready_jobs = 0
        nb_finished_jobs = 0
        nb_running_jobs = 0
        ready_jobs = []
        row_count = self.jobTable.rowCount()
        job_id_col = self.header_labels.index('job id')
        tag_id_col = self.header_labels.index('tag id')
        if self.settings.compress_raw_data:
            compression_col = self.header_labels.index('compression')
        hit_finding_col = self.header_labels.index('hit finding')
        peak2cxi_col = self.header_labels.index('peak2cxi')
        for i in range(row_count):
            # compression jobs
            if self.settings.compress_raw_data:
                compression = self.jobTable.item(i, compression_col).text()
                if compression == 'done':
                    nb_finished_jobs += 1
                elif compression == 'ready':
                    nb_ready_jobs += 1
                    job_id = self.jobTable.item(i, job_id_col).text()
                    ready_job = Job(job_type='compression',
                                    settings=self.settings,
                                    job_id=job_id,
                                    tag_id='NA')
                    ready_jobs.append(ready_job)
                elif compression == 'not ready':
                    nb_not_ready_jobs += 1
                else:
                    nb_running_jobs += 1
                nb_total_jobs += 1
            # hit finding jobs
            hit_finding = self.jobTable.item(i, hit_finding_col).text()
            if hit_finding == 'done':
                nb_finished_jobs += 1
            elif hit_finding == 'ready':
                nb_ready_jobs += 1
                job_id = self.jobTable.item(i, job_id_col).text()
                ready_job = Job(job_type='hit finding',
                                settings=self.settings,
                                job_id=job_id,
                                tag_id=self.hit_tag,
                                compressed=self.settings.compress_raw_data)
                ready_jobs.append(ready_job)
            elif hit_finding == 'not ready':
                nb_not_ready_jobs += 1
            else:
                nb_running_jobs += 1
            nb_total_jobs += 1
            # peak2cxi jobs
            peak2cxi = self.jobTable.item(i, peak2cxi_col).text()
            if peak2cxi == 'done':
                nb_finished_jobs += 1
            elif peak2cxi == 'ready':
                nb_ready_jobs += 1
                job_id = self.jobTable.item(i, job_id_col).text()
                tag_id = self.jobTable.item(i, tag_id_col).text()
                ready_job = Job(job_type='peak2cxi',
                                settings=self.settings,
                                job_id=job_id,
                                tag_id=tag_id)
                ready_jobs.append(ready_job)
            elif peak2cxi == 'not ready':
                nb_not_ready_jobs += 1
            else:
                nb_running_jobs += 1
            nb_total_jobs += 1
        for i in range(min(nb_ready_jobs,
                           self.settings.job_pool_size - nb_running_jobs)):
            job = ready_jobs[i]
            job_existed = False
            for j in range(len(self.jobs)):
                if job == self.jobs[j]:
                    job_existed = True
            if not job_existed:
                print('submitting job of ', str(job))
                self.jobs.append(job)
                job.submit()
                time.sleep(1.0)  # take a break
        print('%d running jobs' % nb_running_jobs)
        print('%d ready jobs' % nb_ready_jobs)
        print('%d finished jobs' % nb_finished_jobs)
        print('%d jobs to to' % (nb_not_ready_jobs + nb_ready_jobs))

    def resizeEvent(self, event):
        width = self.jobTable.width()
        col_count = self.jobTable.columnCount()
        header = self.jobTable.horizontalHeader()

        for i in range(col_count):
            header.resizeSection(i, width // col_count)

    def closeEvent(self, _):
        self.timer.stop()

    def export_jobs(self):
        nb_cols = self.jobTable.horizontalHeader().count()
        header_labels = []
        for i in range(nb_cols):
            header_labels.append(
                self.jobTable.horizontalHeaderItem(i).text()
            )
        nb_rows = self.jobTable.rowCount()
        rows = []
        for i in range(nb_rows):
            row = []
            for j in range(nb_cols):
                row.append(self.jobTable.item(i, j).text())
            rows.append(row)
        with open('click.csv', 'w') as f:
            writer = csv.writer(f)
            print(header_labels)
            writer.writerow(header_labels)
            writer.writerows(rows)


class Job(object):
    def __init__(self, job_type, settings, job_id, tag_id, **kwargs):
        self.job_type = job_type
        self.settings = settings
        self.job_id = job_id
        self.tag_id = tag_id
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.job_thread = None

    def submit(self):
        if self.job_type == 'compression':
            self.job_thread = CompressorThread(self.job_id, self.settings)
        elif self.job_type == 'hit finding':
            self.job_thread = HitFinderThread(
                self.settings,
                job=self.job_id,
                tag=self.tag_id,
                compressed=self.compressed,
            )
        elif self.job_type == 'peak2cxi':
            self.job_thread = Peak2CxiThread(
                self.settings,
                job=self.job_id,
                tag=self.tag_id,
            )
        self.job_thread.start()

    def __cmp__(self, other):
        return __cmp__((self.job_id, self.tag_id, self.job_type),
                       (other.job_id, other.tag_id, other.job_type))

    def __lt__(self, other):
        tuple1 = (self.job_id, self.tag_id, self.job_type)
        tuple2 = (other.job_id, other.tag_id, other.job_type)
        return tuple1 < tuple2

    def __eq__(self, other):
        tuple1 = (self.job_id, self.tag_id, self.job_type)
        tuple2 = (other.job_id, other.tag_id, other.job_type)
        return tuple1 == tuple2

    def __gt__(self, other):
        tuple1 = (self.job_id, self.tag_id, self.job_type)
        tuple2 = (other.job_id, other.tag_id, other.job_type)
        return tuple1 > tuple2

    def __str__(self):
        return '%s:%s-%s' % (self.job_type, self.job_id, self.tag_id)


def check_cxi_comp(cxi_comp_dir):
    stat = None
    if os.path.isdir(cxi_comp_dir):
        stat_file = os.path.join(cxi_comp_dir, 'stat.yml')
        if os.path.exists(stat_file):
            with open(stat_file, 'r') as f:
                stat = yaml.load(f)
    if stat is None:
        stat = {}
    return stat


def check_cxi_hit(hit_dir):
    stat_file = os.path.join(hit_dir, 'stat.yml')
    stat = None
    if os.path.exists(stat_file):
        with open(stat_file, 'r') as f:
            stat = yaml.load(f)
    if stat is None:
        stat = {}
    return stat


def check_peak2cxi(tag_dir):
    stat_file = os.path.join(tag_dir, 'peak2cxi.yml')
    stat = None
    if os.path.exists(stat_file):
        with open(stat_file) as f:
            stat = yaml.load(f)
    if stat is None:
        stat = {}
    return stat
