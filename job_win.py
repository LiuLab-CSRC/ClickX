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

        # batch jobs
        self.curr_jobs = []

        # job window
        self.browse_btn.clicked.connect(self.select_workdir)
        self.crawler_start_btn.clicked.connect(self.start_crawler)
        self.crawler_stop_btn.clicked.connect(self.stop_crawler)
        self.job_table.customContextMenuRequested.connect(self.show_job_menu)

    # job window
    @pyqtSlot()
    def select_workdir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Select a directory')
        self.workdir.setText(directory)

    @pyqtSlot()
    def start_crawler(self):
        self.crawler_start_btn.setEnabled(False)
        workdir = self.workdir.text()
        self.crawler_thread = CrawlerThread(workdir=workdir)
        self.crawler_thread.jobs.connect(self.update_jobs)
        self.crawler_thread.start()
        self.crawler_stop_btn.setEnabled(True)

    @pyqtSlot()
    def stop_crawler(self):
        self.crawler_start_btn.setEnabled(True)
        self.crawler_thread.terminate()
        self.crawler_stop_btn.setEnabled(False)

    @pyqtSlot(list)
    def update_jobs(self, jobs):
        job_table = self.job_table
        print('%d jobs found' % len(jobs))
        curr_job_ids = [job['id'] for job in self.curr_jobs]
        print(curr_job_ids)
        row_count = job_table.rowCount()
        for i in range(len(jobs)):
            job = jobs[i]
            if job['id'] in curr_job_ids:
                continue
            job_table.insertRow(row_count + i)
            item = QTableWidgetItem(job['name'])
            job_table.setItem(i, 0, item)
            self.curr_jobs.append(job)

    @pyqtSlot(QPoint)
    def show_job_menu(self, pos):
        job_table = self.job_table
        menu = QMenu()
        row = job_table.currentRow()
        job = job_table.item(row, 0)
        action_h52cxi = menu.addAction('run h52cxi')
        action_hit_finding = menu.addAction('run hit finding')
        action = menu.exec_(job_table.mapToGlobal(pos))
        if action == action_h52cxi:
            print('run h52cxi')
        elif action == action_hit_finding:
            print('run hit finding')