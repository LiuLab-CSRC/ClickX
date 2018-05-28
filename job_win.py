from PyQt5.QtCore import pyqtSlot, QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QMenu
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.uic import loadUi

from threads import *


class JobWindow(QWidget):
    view_hits = pyqtSignal(str, str)

    def __init__(self, settings, main_win):
        super(JobWindow, self).__init__()
        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/job_win.ui' % dir_, self)
        self.frame.adjustSize()
        self.settings = settings
        self.main_win = main_win
        self.workdir = settings.workdir
        self.raw_dataset = settings.raw_dataset
        self.compressed_dataset = settings.compressed_dataset
        self.rawDataset.setText(self.raw_dataset)
        self.compressedDataset.setText(self.compressed_dataset)

        self.header_labels = settings.table_columns
        self.jobTable.setColumnCount(len(self.header_labels))
        self.jobTable.setHorizontalHeaderLabels(self.header_labels)
        self.compressed_batch_size = settings.compressed_batch_size
        self.compressed_datatype = settings.compressed_datatype

        self.curr_conf = []
        self.crawler_running = False

        # threads
        self.crawler_thread = None
        self.compressor_threads = []
        self.hit_finder_threads = []
        self.peak2cxi_threads = []

        # slots
        self.crawlerButton.clicked.connect(self.start_or_stop_crawler)
        self.jobTable.customContextMenuRequested.connect(self.show_job_menu)

    def update_info(self, settings):
        self.settings = settings
        self.workdir = settings.workdir
        self.raw_dataset = settings.raw_dataset
        self.compressed_dataset = settings.compressed_dataset
        self.rawDataset.setText(self.raw_dataset)
        self.compressedDataset.setText(self.compressed_dataset)
        self.header_labels = settings.table_columns
        self.compressed_batch_size = settings.compressed_batch_size
        self.compressed_datatype = settings.compressed_datatype
        self.rawDataset.setText(self.raw_dataset)
        self.compressedDataset.setText(self.compressed_dataset)
        self.jobTable.setColumnCount(len(self.header_labels))
        self.jobTable.setHorizontalHeaderLabels(self.header_labels)

    @pyqtSlot()
    def start_or_stop_crawler(self):
        if self.crawler_running:
            self.crawler_running = False
            self.crawlerButton.setText('Start Crawler')
            self.crawler_thread.terminate()
        else:
            self.crawler_running = True
            self.crawlerButton.setText('Stop Crawler')
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
            self.hitFindingConf.addItem(tag, userData=conf)
            self.curr_conf.append(conf)

    @pyqtSlot(dict)
    def update_stat(self, stat):
        total_raw_frames = stat.get('total raw frames', 0)
        self.rawFrames.setText(str(total_raw_frames))
        curr_id = self.hitFindingConf.currentIndex()
        tag = self.hitFindingConf.itemText(curr_id)
        total_processed_frame_dict = stat.get('total processed frames', {})
        total_processed_hits_dict = stat.get('total processed hits', {})
        if tag in stat['total processed frames'].keys():
            total_processed_hits = total_processed_hits_dict.get(tag, 0)
            total_processed_frames = total_processed_frame_dict.get(tag, 0)
            if total_processed_frames != 0:
                total_hit_rate = float(total_processed_hits) /\
                                 total_processed_frames * 100.
            else:
                total_hit_rate = 0.
            self.processedHits.setText(str(total_processed_hits))
            self.processedFrames.setText(str(total_processed_frames))
            self.hitRate.setText('%.2f%%' % total_hit_rate)

    @pyqtSlot(QPoint)
    def show_job_menu(self, pos):
        menu = QMenu()
        items = self.jobTable.selectedItems()
        rows = []
        for item in items:
            rows.append(self.jobTable.row(item))
        rows = set(rows)  # remove duplicates
        action_compression = menu.addAction('run compressor')
        action_hit_finding = menu.addAction('run hit finder')
        action_peak2cxi = menu.addAction('convert peaks to cxi')
        menu.addSeparator()
        action_view_hits = menu.addAction('view hits')
        menu.addSeparator()
        action_sum = menu.addAction('calc sum')
        action = menu.exec_(self.jobTable.mapToGlobal(pos))

        if action == action_compression:
            jobs = []
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')).text()
                jobs.append(job_id)
            for job in jobs:
                compressor_thread = CompressorThread(job, self.settings)
                self.compressor_threads.append(compressor_thread)
                compressor_thread.start()
        elif action == action_hit_finding:
            curr_id = self.hitFindingConf.currentIndex()
            if curr_id == -1:
                print('No valid conf available!')
                return
            hit_conf = self.hitFindingConf.itemData(curr_id)
            hit_tag = self.hitFindingConf.itemText(curr_id)
            for row in rows:
                job = self.jobTable.item(
                    row, self.header_labels.index('job id')
                ).text()
                hit_finder_thread = HitFinderThread(
                    self.settings,
                    job=job,
                    conf=hit_conf,
                    tag=hit_tag,
                )
                self.hit_finder_threads.append(hit_finder_thread)
                hit_finder_thread.start()
        elif action == action_peak2cxi:
            jobs = []
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')).text()
                tag_id = self.jobTable.item(
                    row, self.header_labels.index('tag id')).text()
            jobs.append([job_id, tag_id])
            for job in jobs:
                peak2cxi_thread = Peak2CxiThread(
                    self.settings,
                    job=job[0],
                    tag=job[1],
                )
                self.peak2cxi_threads.append(peak2cxi_thread)
                peak2cxi_thread.start()
        elif action == action_view_hits:
            row = self.jobTable.currentRow()
            job = self.jobTable.item(
                row, self.header_labels.index('job id')).text()
            tag = self.jobTable.item(
                row, self.header_labels.index('tag id')).text()
            self.view_hits.emit(job, tag)
        # elif action == action_sum:
        #     s = 0
        #     for item in items:
        #         try:
        #             s += int(item.text())
        #         except ValueError:
        #             print('%s not a number' % item.text())
        #     print('sum of selected items: %.2f' % s)

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
