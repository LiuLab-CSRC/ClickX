#!/bin/env python
# -*- coding: utf-8 -*-


"""Generate cxi files from hits in peak files using MPI.

Usage:
   batch_peak2cxi.py <peak-file> <hit-dir> [options]

Options:
    -h --help                   Show this screen.
    --min-peaks NUM             Specify min peaks for a hit [default: 20].
    --cxi-size SIZE             Specify max frame in one single cxi file
                                [default: 100].
    --raw-data-path PATH        Specify path of raw data [default: data].
    --peak-info-path PATH       Specify path of peak info [default: peak_info].
    --cxi-dtype DATATYPE        Specify datatype of patterns in compressed cxi
                                file [default: auto].
    --mask-file FILE            Specify mask file for cxi.
    --extra-datasets DATASETS   Specify extra datasets of cheetah.
    --shuffle SHUFFLE           Whether to use shuffle filter in compression
                                [default: True].
    --batch-size SIZE           Specify batch size in a job [default: 10].
    --buffer-size SIZE          Specify buffer size in MPI communication
                                [default: 500000].
    --update-freq FREQ          Specify update frequency of progress
                                [default: 10].
    --flush                     Flush output of print.

"""
from __future__ import print_function
from six import print_ as print
from docopt import docopt
import sys
import os
import time

from mpi4py import MPI
import numpy as np
import yaml
import util


def master_run(args):
    flush = args['--flush']
    # mkdir if not exist
    hit_dir = args['<hit-dir>']
    if not os.path.isdir(hit_dir):
        os.mkdir(hit_dir)
    peak_file = args['<peak-file>']
    peak_info = np.load(peak_file)

    batch_size = int(args['--batch-size'])
    nb_jobs = int(np.ceil(len(peak_info) / batch_size))

    # collect jobs
    ids = np.array_split(np.arange(len(peak_info)), nb_jobs)
    jobs = []
    for i in range(len(ids)):
        jobs.append(peak_info[ids[i]])
    print('%d jobs, %d frames to be processed' %
          (nb_jobs, len(peak_info)), flush=flush)

    # other parameters
    buffer_size = int(args['--buffer-size'])
    update_freq = int(args['--update-freq'])

    # dispatch jobs
    job_id = 0
    reqs = {}
    workers = set(range(1, size))
    finished_workers = set()
    time_start = time.time()
    for worker in workers:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(job, dest=worker)
        reqs[worker] = comm.irecv(buf=buffer_size, source=worker)
        print('job %d/%d --> slave %d'
              % (job_id, nb_jobs, worker), flush=flush)
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        workers -= finished_workers
        time.sleep(0.1)  # take a break
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                if job_id < nb_jobs:
                    print('job %d/%d --> slave %d' %
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
        if job_id % update_freq == 0:
            # update stat
            progress = float(job_id) / nb_jobs * 100
            stat_dict = {
                'progress': '%.2f%%' % progress,
                'duration/sec': 'not finished',
                'total jobs': nb_jobs,
            }
            stat_file = os.path.join(hit_dir, 'peak2cxi.yml')
            with open(stat_file, 'w') as f:
                yaml.dump(stat_dict, f, default_flow_style=False)

    all_done = False
    while not all_done:
        time.sleep(0.1)
        workers -= finished_workers
        all_done = True
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_done = False
    time_end = time.time()
    duration = time_end - time_start

    stat_dict = {
        'progress': 'done',
        'duration/sec': duration,
        'total jobs': nb_jobs,
    }
    stat_file = os.path.join(hit_dir, 'peak2cxi.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    print('All Done!', flush=flush)
    MPI.Finalize()


def worker_run(args):
    stop = False
    peak_file = args['<peak-file>']
    hit_dir = args['<hit-dir>']
    min_peaks = int(args['--min-peaks'])
    cxi_size = int(args['--cxi-size'])
    raw_data_path = args['--raw-data-path']
    peak_info_path = args['--peak-info-path']
    cxi_dtype = args['--cxi-dtype']
    extra_datasets = args['--extra-datasets']
    mask_file = args['--mask-file']
    buffer_size = int(args['--buffer-size'])
    prefix = os.path.basename(peak_file).split('.')[0]
    if args['--shuffle'] == 'True':
        shuffle = True
    else:
        shuffle = False

    # perform compression
    batch = []
    count = 0
    while not stop:
        job = comm.recv(buf=buffer_size, source=0)
        for i in range(len(job)):
            if job[i]['nb_peak'] >= min_peaks:
                batch.append(job[i])
            if len(batch) == cxi_size:
                # save full cxi file
                cxi_file = os.path.join(
                    hit_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
                )
                util.save_full_cxi(
                    batch, cxi_file,
                    raw_data_path=raw_data_path,
                    peak_info_path=peak_info_path,
                    mask_file=mask_file,
                    extra_datasets=extra_datasets,
                    cxi_dtype=cxi_dtype,
                    shuffle=shuffle,
                )
                sys.stdout.flush()
                batch.clear()
                count += 1
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # save last cxi batch if not empty
    if len(batch) > 0:
        # save full cxi
        cxi_file = os.path.join(
            hit_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
        )
        util.save_full_cxi(
            batch, cxi_file,
            raw_data_path=raw_data_path,
            peak_info_path=peak_info_path,
            mask_file=mask_file,
            extra_datasets=extra_datasets,
            cxi_dtype=cxi_dtype,
            shuffle=shuffle
        )
        sys.stdout.flush()
    done = True
    comm.send(done, dest=0)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch peak2cxi with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        worker_run(args)
