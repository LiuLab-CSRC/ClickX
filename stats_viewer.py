# -*- coding: utf-8 -*-


from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import os

import numpy as np
from functools import partial


PRIMARY_LABEL_STYLE = {
    'color': 'green',
    'font-size': '20pt'
}

SECONDARY_LABEL_STYLE = {
    'color': 'yellow',
    'font-size': '20pt'
}


class StatsViewer(QWidget):
    def __init__(self, settings, main_win):
        super(StatsViewer, self).__init__()
        # setup layout
        dir_ = os.path.abspath(os.path.dirname(__file__))
        loadUi('%s/ui/window/hits.ui' % dir_, self)
        self.plotWidget.setBackground(QColor(80, 80, 80))
        self.primaryData.clear()
        self.secondaryData.clear()
        # common variable
        self.sorted_idx = None
        # plot items
        self.primary_plot, self.secondary_plot = self.create_plot_items()
        self.primary_plot_item = pg.PlotDataItem(
            symbol='o', pen=QColor('green'), symbolBrush=QColor('green')
        )
        self.primary_hist_item = pg.PlotDataItem(
            stepMode=True, fillLevel=0,
            pen=QColor('green'), fillBrush=QColor('green')
        )
        self.secondary_plot_item = pg.PlotDataItem(
            symbol='o', pen=QColor('yellow'), symbolBrush=QColor('yellow')
        )
        # load settings
        self.settings = settings
        self.main_win = main_win
        self.data_dict = {}

        self.browseButton.clicked.connect(self.choose_and_load_hits)
        self.plotButton.clicked.connect(self.plot)
        self.primary_plot.vb.sigResized.connect(self.update_views)
        self.primary_plot_item.sigPointsClicked.connect(self.view_event)
        self.secondary_plot_item.sigPointsClicked.connect(self.view_event)

    def create_plot_items(self):
        p1 = self.plotWidget.plotItem
        p2 = pg.ViewBox()
        p1.showAxis('right')
        p1.scene().addItem(p2)
        p1.getAxis('right').linkToView(p2)
        p2.setXLink(p1)
        return p1, p2

    @pyqtSlot()
    def choose_and_load_hits(self):
        stats_file, _ = QFileDialog.getOpenFileName(
            self, "Open Stats File", 'cxi_hit', "(*.npy)"
        )
        if len(stats_file) == 0:
            return
        self.statsFile.setText(stats_file)
        self.load_stats(stats_file)

    @pyqtSlot(int, int)
    def view_hits(self, row, _):
        path = self.table.item(row, 0).text()
        dataset = self.table.item(row, 1).text()
        frame = int(self.table.item(row, 2).text())
        self.main_win.maybe_add_file(path)
        self.main_win.load_data(path, dataset=dataset, frame=frame)
        self.main_win.update_file_info()
        self.main_win.change_image()
        self.main_win.update_display()

    @pyqtSlot(object, object)
    def view_event(self, _, points):
        x = points[0].pos()[0]
        event = self.sorted_idx[int(x)]
        print('viewing event %d' % event)
        path = self.data_dict['filepath'][event]
        dataset = self.data_dict['dataset'][event]
        frame = self.data_dict['frame'][event]
        self.main_win.maybe_add_file(path)
        self.main_win.load_data(path, dataset=dataset, frame=frame)
        self.main_win.update_file_info()
        self.main_win.change_image()
        self.main_win.update_display()

    def load_stats(self, stats_file):
        if not os.path.exists(stats_file):
            return
        data = np.load(stats_file)

        # collect all scalar fields for plot
        all_fields = []
        for i in range(len(data)):
            all_fields += list(data[i]['data_dict'])
        all_fields = list(set(all_fields))
        scalar_fields = list(
            set(all_fields) & {
                'total_intensity',
                'max_intensity',
                'clen',
                'fiducial',
                'photon_energy',
                'flow_rate',
                'pressure'
            }
        )

        # collect all data to data dict
        self.data_dict = {
            'filepath': [],
            'dataset': [],
            'frame': [],
            'nb_peak': []
        }
        for field in scalar_fields:
            self.data_dict[field] = []
        for i in range(len(data)):
            self.data_dict['filepath'].append(data[i]['filepath'])
            self.data_dict['dataset'].append(data[i]['dataset'])
            self.data_dict['frame'].append(data[i]['frame'])
            self.data_dict['nb_peak'].append(data[i]['nb_peak'])
            for field in scalar_fields:
                self.data_dict[field].append(data[i]['data_dict'][field])
        for key, value in self.data_dict.items():
            self.data_dict[key] = np.array(value)

        self.primaryData.addItems(scalar_fields + ['nb_peak'])
        self.secondaryData.addItems([''] + scalar_fields + ['nb_peak'])

    def update_views(self):
        self.secondary_plot.setGeometry(
            self.primary_plot.vb.sceneBoundingRect()
        )
        self.secondary_plot.linkedViewChanged(
            self.primary_plot.vb, self.secondary_plot.XAxis
        )

    def plot(self):
        self.primary_plot.clear()
        self.secondary_plot.clear()

        primary_dataset = self.primaryData.currentText()
        primary_data = self.data_dict[primary_dataset]

        if self.linePlot.isChecked():
            sort_data = self.sortDataset.isChecked()
            if sort_data:
                self.sorted_idx = np.argsort(primary_data)
            else:
                self.sorted_idx = np.arange(len(primary_data))

            self.primary_plot_item.setData(
                x=np.arange(len(primary_data)),
                y=primary_data[self.sorted_idx]
            )
            self.primary_plot.addItem(self.primary_plot_item)
            self.primary_plot.getAxis('left').setLabel(
                primary_dataset, **PRIMARY_LABEL_STYLE
            )
            self.primary_plot.getAxis('bottom').setLabel(
                'index', color='#ffffff'
            )
            self.primary_plot.autoRange()

            secondary_dataset = self.secondaryData.currentText()
            if len(secondary_dataset) == 0:
                return
            secondary_data = self.data_dict[secondary_dataset]
            self.secondary_plot_item.setData(
                x=np.arange(len(secondary_data)),
                y=secondary_data[self.sorted_idx]
            )
            self.secondary_plot.addItem(self.secondary_plot_item)
            self.primary_plot.getAxis('right').setLabel(
                secondary_dataset, **SECONDARY_LABEL_STYLE
            )
            self.primary_plot.autoRange()
            self.update_views()

        else:
            if primary_data.max() == primary_data.min():
                print('Skip single value distribution of %.3e'
                      % primary_data[0])
                return
            bin_size = self.binSize.value()
            bin_num = (primary_data.max() - primary_data.min()) / bin_size
            if bin_num > 1000:
                bin_size = (primary_data.max() - primary_data.min()) / 1000
                print('Bin size too small, set to %.2f' % bin_size)
            elif bin_num < 2:
                bin_size = max(
                    (primary_data.max() - primary_data.min()) / 2,
                    0.001
                )
                print('Bin size too big, set to %.2f' % bin_size)

            self.binSize.setValue(bin_size)
            y, x = np.histogram(
                primary_data,
                bins=np.arange(
                    primary_data.min() * 0.9,
                    primary_data.max() * 1.1,
                    bin_size
                )
            )
            self.primary_hist_item.setData(
                x, y
            )
            self.primary_plot.addItem(self.primary_hist_item)
            self.primary_plot.getAxis('bottom').setLabel(
                primary_dataset, **PRIMARY_LABEL_STYLE
            )
            self.primary_plot.getAxis('left').setLabel(
                'count', color='#ffffff'
            )
            self.primary_plot.autoRange()

