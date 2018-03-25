#!/bin/env python

"""Run hit finding on multiple cores using MPI.

Usage:
   batch_hit_finding.py <list-file> <conf-file> <hit-dir> [options]

Options:
    -h --help               Show this screen.
    --batch-size SIZE       Specify batch size in a job [default: 50].
    --buffer-size SIZE      Specify buffer size in MPI communication
                            [default: 500000].
    --update-freq FREQ      Specify update frequency of progress [default: 10].
"""
from mpi4py import MPI
import numpy as np
import pandas as pd
import h5py
import time

import sys
import os
from docopt import docopt
import yaml

from util import find_peaks, read_image

import warnings
warnings.filterwarnings("ignore")


def collect_jobs(files, dataset, batch_size):
    jobs = []
    batch = []
    total_frame = 0
    for f in files:
        try:
            shape = h5py.File(f, 'r')[dataset].shape
            if len(shape) == 3:
                nb_frame = shape[0]
                for i in range(nb_frame):
                    batch.append({'filepath': f, 'frame': i})
                    total_frame += 1
                    if len(batch) == batch_size:
                        jobs.append(batch)
                        batch = []
            else:
                batch.append({'filepath': f, 'frame': 0})
                total_frame += 1
                if len(batch) == batch_size:
                    jobs.append(batch)
                    batch = []
        except OSError:
            pass
    if len(batch) > 0:
        jobs.append(batch)
    return jobs, total_frame


def master_run(args):
    hit_dir = args['<hit-dir>']
    if not os.path.isdir(hit_dir):
        os.makedirs(hit_dir)
    cxi_lst = args['<list-file>']
    with open(cxi_lst) as f:
        _files = f.readlines()
    # remove trailing '/n'
    files = []
    for f in _files:
        files.append(f[:-1])
    # load hit finding configuration file
    with open(args['<conf-file>']) as f:
        conf = yaml.load(f)
    # collect jobs
    dataset = conf['dataset']
    min_peak = conf['min peak num']
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    jobs, total_frame = collect_jobs(files, dataset, batch_size)
    total_jobs = len(jobs)
    print('%d frames, %d jobs to be processed' % (total_frame, total_jobs))

    update_freq = int(args['--update-freq'])
    cxi_prefix = os.path.basename(cxi_lst).split('.')[0]

    # distribute jobs
    job_id = 0
    reqs = {}
    results = []
    slaves = list(range(1, size))
    progress_file = os.path.join(hit_dir, 'progress.txt')
    time_start = time.time()
    for slave in slaves:
        comm.isend(jobs[job_id], dest=slave)
        reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
        print('job %d/%d sent to %d' % (job_id, total_jobs, slave))
        sys.stdout.flush()
        job_id += 1
    while job_id < total_jobs:
        stop = False
        time.sleep(0.1)  # take a break
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                results += result
                if job_id < total_jobs:
                    print('job %d/%d sent to %d' % (job_id, total_jobs, slave))
                    sys.stdout.flush()
                    comm.isend(stop, dest=slave)
                    comm.isend(jobs[job_id], dest=slave)
                    reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=slave)
                    print('stop signal sent to %d' % slave)
        if job_id % update_freq == 0:
            progress = float(job_id) / total_jobs * 100
            with open(progress_file, 'w') as f:
                f.write(str(progress))

    all_done = False
    while not all_done:
        all_done = True
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                results += result
                slaves.remove(slave)
                stop = True
                comm.isend(stop, dest=slave)
            else:
                all_done = False
    time_end = time.time()
    duration = time_end - time_start
    
    # save results
    with open(progress_file, 'w') as f:
        f.write('done')

    csv_file = os.path.join(hit_dir, '%s.csv' % cxi_prefix)
    df = pd.DataFrame(results)
    df.to_csv(csv_file)

    total_hits = len(df['nb_peak'] > min_peak)
    total_frame = len(df)
    hit_rate = float(total_hits) / total_frame * 100.
    stat_dict = {
        'duration/sec': duration,
        'total frames': len(results),
        'total jobs': total_jobs,
        'total hits': total_hits,
        'hit rate': hit_rate,
    }
    stat_file = os.path.join(hit_dir, 'stat.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    print('All Done!')


def slave_run(args):
    stop = False
    filepath = None
    h5_obj = None
    buffer_size = int(args['--buffer-size'])

    # hit finding parameters
    with open(args['<conf-file>']) as f:
        conf = yaml.load(f)
    gaussian_sigma = conf['gaussian filter sigma']
    mask_file = conf['mask file']
    if mask_file is not None:
        mask = read_image(mask_file)
    else:
        mask = None
    max_pean_num = conf['max peak num']
    min_peak_num = conf['min peak num']
    min_distance = conf['min distance']
    min_gradient = conf['min gradient']
    min_snr = conf['min snr']
    dataset = conf['dataset']

    # perform hit finding
    while not stop:
        job = comm.recv(buf=buffer_size, source=0)
        for i in range(len(job)):
            _filepath = job[i]['filepath']
            frame = job[i]['frame']
            if _filepath != filepath:
                filepath = _filepath
                h5_obj = h5py.File(filepath, 'r')
            image = read_image(filepath, frame=frame,
                               h5_obj=h5_obj, dataset=dataset)
            peaks_dict = find_peaks(image, mask=mask,
                                    gaussian_sigma=gaussian_sigma,
                                    min_distance=min_distance,
                                    min_gradient=min_gradient,
                                    max_peaks=max_pean_num,
                                    min_snr=min_snr)
            print(peaks_dict)
            job[i]['nb_peak'] = len(peaks_dict['strong'])
        comm.send(job, dest=0)
        stop = comm.recv(source=0)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch hit finding with at leat 2 processs!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        slave_run(args)
