from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QPoint, Qt, QTimer
from PyQt5.QtWidgets import QWidget, QMenu, QTableWidgetItem
from PyQt5.uic import loadUi
from threads import *


class JobWindow(QWidget):
    view_hits = pyqtSignal(str, str)

    def __init__(self, settings, main_win):
        super(JobWindow, self).__init__()
        # setup ui
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/job_win.ui' % dir_, self)
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

        self.jobs_info = None
        self.conf_files = None
        self.jobs_stat = None
        self.auto_submit = False
        self.curr_conf = []
        self.jobs = []

        self.timer = QTimer()

        # slots
        self.jobTable.customContextMenuRequested.connect(self.show_job_menu)
        self.autoSubmit.toggled.connect(self.change_auto_submit)
        self.timer.timeout.connect(self.check_and_submit_jobs)

    def start(self):
        self.timer.start(5000)

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

    def crawler_run(self):
        # check data from h5 lst
        jobs_info = []
        raw_lst_files = glob('%s/raw_lst/*.lst' % self.workdir)
        total_raw_frames = 0
        total_processed_frames = {}
        total_processed_hits = {}
        for raw_lst in raw_lst_files:
            time1 = os.path.getmtime(raw_lst)
            job_name = os.path.basename(raw_lst).split('.')[0]
            # check compression status
            compression = 'ready'
            comp_ratio = 0
            raw_frames = 0
            cxi_comp_dir = os.path.join(self.workdir, 'cxi_comp', job_name)
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
            time2 = np.inf
            peak2cxi = 'not ready'
            processed_frames = 0
            processed_hits = 0
            hit_rate = 0.
            cxi_hit_dir = os.path.join(self.workdir, 'cxi_hit', job_name)
            hit_tags = glob('%s/*' % cxi_hit_dir)
            tags = [os.path.basename(tag) for tag in hit_tags]
            if len(hit_tags) == 0:
                jobs_info.append(
                    {
                        'job id': job_name,
                        'raw frames': raw_frames,
                        'compression progress': compression,
                        'compression ratio': comp_ratio,
                        'tag id': 'NA',  # dummy tag
                        'hit finding progress': hit_finding,
                        'processed frames': processed_frames,
                        'processed hits': processed_hits,
                        'hit rate': hit_rate,
                        'peak2cxi progress': peak2cxi,
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
                        time2 = stat.get('time start', np.inf)
                        processed_hits = stat.get('processed hits', 0)
                        processed_frames = stat.get('processed frames', 0)
                        hit_finding = stat.get('progress', 0)
                        if str(hit_finding) == 'done':
                            peak2cxi = 'ready'
                            cxi_files = glob('%s/*.cxi' % tag_dir)
                            if len(cxi_files) > 0:
                                peak2cxi = 'done'
                        else:
                            peak2cxi = 'not ready'
                        if tag in total_processed_hits:
                            total_processed_hits[tag] += processed_hits
                        else:
                            total_processed_hits[tag] = processed_hits
                        if tag in total_processed_frames:
                            total_processed_frames[tag] += processed_frames
                        else:
                            total_processed_frames[tag] = processed_frames
                        hit_rate = stat.get('hit rate')
                    jobs_info.append(
                        {
                            'job id': job_name,
                            'raw frames': raw_frames,
                            'compression progress': compression,
                            'compression ratio': comp_ratio,
                            'tag id': tag,
                            'hit finding progress': hit_finding,
                            'processed hits': processed_hits,
                            'processed frames': processed_frames,
                            'hit rate': hit_rate,
                            'peak2cxi progress': peak2cxi,
                            'time1': time1,
                            'time2': time2,
                        }
                    )
        self.jobs_info = sorted(
            jobs_info, key=operator.itemgetter('job id', 'time2')
        )

        # check hit finding conf files
        conf_dir = os.path.join(self.workdir, 'conf')
        self.conf_files = glob('%s/*.yml' % conf_dir)

        # check stat
        self.jobs_stat = {
            'total raw frames': total_raw_frames,
            'total processed frames': total_processed_frames,
            'total processed hits': total_processed_hits,
        }

    @pyqtSlot(list)
    def update_jobs_info(self):
        for i,job_info in enumerate(self.jobs_info):
            self.fill_table_row(job_info, i)

    @pyqtSlot(list)
    def update_conf(self):
        for conf in self.conf_files:
            if conf in self.curr_conf:
                continue
            tag = os.path.basename(conf).split('.')[0]
            self.hitFindingConf.addItem(tag, userData=conf)
            self.curr_conf.append(conf)

    @pyqtSlot(dict)
    def update_stat(self):
        self.rawFrames.setText(
            str(self.jobs_stat.get('total raw frames', 0)))
        curr_id = self.hitFindingConf.currentIndex()
        tag = self.hitFindingConf.itemText(curr_id)
        total_processed_frame_dict = self.jobs_stat.get(
            'total processed frames', {})
        total_processed_hits_dict = self.jobs_stat.get(
            'total processed hits', {})
        if tag in self.jobs_stat['total processed frames']:
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
            curr_id = self.hitFindingConf.currentIndex()
            if curr_id == -1:
                print('No valid conf available!')
                return
            hit_conf = self.hitFindingConf.itemData(curr_id)
            hit_tag = self.hitFindingConf.itemText(curr_id)
            for row in rows:
                job_id = self.jobTable.item(
                    row, self.header_labels.index('job id')
                ).text()
                job = Job(job_type='hit finding',
                          settings=self.settings,
                          job_id=job_id,
                          tag_id=hit_tag,
                          hit_conf=hit_conf)
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
        self.update_conf()
        self.update_stat()
        if self.auto_submit:
            self.find_and_submit_jobs()

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
        compression_col = self.header_labels.index('compression progress')
        hit_finding_col = self.header_labels.index('hit finding progress')
        peak2cxi_col = self.header_labels.index('peak2cxi progress')
        for i in range(row_count):
            # compression jobs
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
            curr_id = self.hitFindingConf.currentIndex()
            if curr_id == -1:
                print('No valid conf available!')
                return
            hit_conf = self.hitFindingConf.itemData(curr_id)
            hit_tag = self.hitFindingConf.itemText(curr_id)
            if hit_finding == 'done':
                nb_finished_jobs += 1
            elif hit_finding == 'ready':
                nb_ready_jobs += 1
                job_id = self.jobTable.item(i, job_id_col).text()
                ready_job = Job(job_type='hit finding',
                                settings=self.settings,
                                job_id=job_id,
                                tag_id=hit_tag,
                                hit_conf=hit_conf)
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
                print('submitting job of ', job)
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

    def closeEvent(self):
        self.timer.stop()


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
                conf=self.hit_conf,
                tag=self.tag_id,
            )
        elif self.job_type == 'peak2cxi':
            self.job_thread = Peak2CxiThread(
                self.settings,
                job=self.job_id,
                tag=self.tag_id,
            )
        self.job_thread.start()

    def __eq__(self, other):
        if self.job_id == other.job_id \
                and self.tag_id == other.tag_id\
                and self.job_type == other.job_type:
            return True
        else:
            return False

    def __str__(self):
        return '%s:%s-%s' % (self.job_type, self.job_id, self.tag_id)