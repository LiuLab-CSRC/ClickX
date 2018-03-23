import sys
import os
from os.path import abspath, dirname

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt, QPoint, QThread
from PyQt5.QtWidgets import QWidget, QDialog, QFileDialog, QMenu
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.uic import loadUi

from threads import *


class JobWindow(QWidget):
    def __init__(self, parent=None):
        super(JobWindow, self).__init__(parent)
        # set ui
        loadUi('jobs.ui', self)
        _dir = dirname(abspath(__file__))
        self.workdir.setText(_dir)

        col_count = self.job_table.columnCount()
        self.header = {
            x: self.job_table.horizontalHeaderItem(x).text() \
            for x in range(col_count)
        }
        self.curr_conf = []
        self.crawler_running = False

        # job window
        self.browse_btn.clicked.connect(self.select_workdir)
        self.crawler_btn.clicked.connect(self.start_or_stop_crawler)
        self.job_table.customContextMenuRequested.connect(self.show_job_menu)

    # job window
    @pyqtSlot()
    def select_workdir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Select a directory')
        self.workdir.setText(directory)

    @pyqtSlot()
    def start_or_stop_crawler(self):
        if self.crawler_running:
            self.crawler_running = False
            self.crawler_btn.setText('start crawler')
            self.crawler_thread.terminate()
        else:
            self.crawler_running = True
            self.crawler_btn.setText('stop crawler')
            workdir = self.workdir.text()
            self.crawler_thread = CrawlerThread(workdir=workdir)
            self.crawler_thread.jobs.connect(self.update_jobs)
            self.crawler_thread.conf.connect(self.update_conf)
            self.crawler_thread.start()

    @pyqtSlot(list)
    def update_jobs(self, jobs):
        job_table = self.job_table
        row_count = job_table.rowCount()
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

    @pyqtSlot(QPoint)
    def show_job_menu(self, pos):
        job_table = self.job_table
        menu = QMenu()
        row = job_table.currentRow()
        job = job_table.item(row, 0).text()
        workdir = self.workdir.text()
        action_h52cxi = menu.addAction('run h52cxi')
        action_hit_finding = menu.addAction('run hit finding')
        action = menu.exec_(job_table.mapToGlobal(pos))
        if action == action_h52cxi:
            self.conv_thread = ConversionThread(
                workdir=workdir,
                job=job,
                h5_dataset=self.h5_dataset.text(),
                cxi_dataset=self.cxi_dataset.text(),
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
            )
            self.hit_finding_thread.start()

    def fill_table_row(self, row_dict, row):
        row_count = self.job_table.rowCount()
        if row_count == row:
            self.job_table.insertRow(row_count)
        for col, field in self.header.items():
            item = self.job_table.item(row, col)
            if item is None:
                item = QTableWidgetItem(row_dict[field])
                item.setTextAlignment(
                    QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter
                )
                self.job_table.setItem(row, col, item)
            else:
                item.setText(str(row_dict[field]))
