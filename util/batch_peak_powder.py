#!/bin/env python
# -*- coding: utf-8 -*-


"""
Generate peak powder pattern on multiple cores using MPI.

Usage:
   batch_peak_powder.py <file-lst> <conf-file> [options]

Options:
    -h --help               Show this screen.
    -o FILE                 Specify output filename [default: powder.npz].
    --batch-size SIZE       Specify batch size in a job [default: 10].
    --buffer-size SIZE      Specify buffer size in MPI communication
                            [default: 500000].
    --flush                 Flush output of print.
"""
from __future__ import print_function
from six import print_ as print
from mpi4py import MPI
import h5py
import numpy as np
import time

import sys
import os
from docopt import docopt
import yaml
import util


def master_run(args):
    flush = args['--flush']
    file_lst = args['<file-lst>']
    with open(file_lst) as f:
        _files = f.readlines()
    # remove trailing '/n'
    files = []
    for f in _files:
        if '\n' == f[-1]:
            files.append(f[:-1])
        else:
            files.append(f)
    # load hit finding configuration file
    with open(args['<conf-file>']) as f:
        conf = yaml.load(f)
    # collect jobs
    dataset = conf['dataset']
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    jobs, nb_frames = util.collect_jobs(files, dataset, batch_size)
    nb_jobs = len(jobs)
    print('%d frames, %d jobs to be processed' %
          (nb_frames, nb_jobs), flush=flush)

    # dispatch jobs
    job_id = 0
    reqs = {}
    peaks = []
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(job, dest=worker)
        reqs[worker] = comm.irecv(buf=buffer_size, source=worker)
        print('job %d/%d  --> %d' % (job_id, nb_jobs, worker), flush=flush)
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        time.sleep(0.1)  # take a break
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                peaks += result
                if job_id < nb_jobs:
                    print('job %d/%d --> %d' %
                          (job_id, nb_jobs, worker), flush=flush)
                    comm.isend(stop, dest=worker)
                    comm.isend(jobs[job_id], dest=worker)
                    reqs[worker] = comm.irecv(buf=buffer_size, source=worker)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=worker)
                    print('stop signal --> %d' % worker, flush=flush)
                    finished_workers.add(worker)

    all_done = False
    while not all_done:
        all_done = True
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                peaks += result
                stop = True
                print('stop signal --> %d' % worker, flush=flush)
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_done = False

    # build and save peak powder
    filepath = jobs[0][0]['filepath']
    frame = jobs[0][0]['frame']
    h5_obj = h5py.File(filepath, 'r')
    image = util.read_image(
        filepath, frame=frame, h5_obj=h5_obj, dataset=dataset)['image']
    powder = np.zeros(image.shape)
    peaks = np.round(np.array(peaks)).astype(np.int)
    powder[peaks[:, 0], peaks[:, 1]] = 1
    powder_file = args['-o']
    dir_ = os.path.dirname(powder_file)
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
    np.savez(powder_file, powder_pattern=powder, powder_peaks=peaks)
    print('All Done!', flush=flush)
    MPI.Finalize()


def worker_run(args):
    stop = False
    filepath = None
    h5_obj = None
    buffer_size = int(args['--buffer-size'])
    flush = args['--flush']

    # hit finding parameters
    with open(args['<conf-file>']) as f:
        conf = yaml.load(f)
    center = conf['center']
    adu_per_photon = conf['adu per photon']
    epsilon = conf['epsilon']
    bin_size = conf['bin size']
    if conf['mask on']:
        mask = util.read_image(conf['mask file'])['image']
    else:
        mask = None
    hit_finder = conf['hit finder']
    gaussian_sigma = conf['gaussian filter sigma']
    min_distance = conf['min distance']
    min_gradient = conf['min gradient']
    max_peaks = conf['max peaks']
    min_snr = conf['min snr']
    min_pixels = conf['min pixels']
    peak_refine_mode = conf['peak refine mode']
    snr_mode = conf['snr mode']
    sig_radius = conf['signal radius']
    bg_inner_radius = conf['background inner radius']
    bg_outer_radius = conf['background outer radius']
    crop_size = conf['crop size']
    bg_ratio = conf['background ratio']
    sig_ratio = conf['signal ratio']
    sig_thres = conf['signal threshold']
    dataset = conf['dataset']

    # perform hit finding
    while not stop:
        job = comm.recv(buf=buffer_size, source=0)
        peaks = []
        for i in range(len(job)):
            _filepath = job[i]['filepath']
            frame = job[i]['frame']
            if _filepath != filepath:
                filepath = _filepath
                h5_obj = h5py.File(filepath, 'r')
            image = util.read_image(filepath, frame=frame,
                                    h5_obj=h5_obj, dataset=dataset)['image']
            peaks_dict = util.find_peaks(
                image, center,
                adu_per_photon=adu_per_photon,
                epsilon=epsilon,
                bin_size=bin_size,
                mask=mask,
                hit_finder=hit_finder,
                gaussian_sigma=gaussian_sigma,
                min_gradient=min_gradient,
                min_distance=min_distance,
                max_peaks=max_peaks,
                min_snr=min_snr,
                min_pixels=min_pixels,
                refine_mode=peak_refine_mode,
                snr_mode=snr_mode,
                signal_radius=sig_radius,
                bg_inner_radius=bg_inner_radius,
                bg_outer_radius=bg_outer_radius,
                crop_size=crop_size,
                bg_ratio=bg_ratio,
                signal_ratio=sig_ratio,
                signal_thres=sig_thres,
            )
            if peaks_dict['strong'] is not None:
                peaks += peaks_dict['strong'].tolist()
        comm.send(peaks, dest=0)
        stop = comm.recv(source=0)
        if stop:
            print('slave %d is exiting' % rank, flush=flush)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch hit finder with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    argv = docopt(__doc__)
    if rank == 0:
        master_run(argv)
    else:
        worker_run(argv)
