#!/bin/env python

"""Convert h5 files to cxi files on multiple cores using MPI.

Usage:
   batch_h52cxi.py <h5-lst> <h5-dataset> <cxi-dir> <cxi-lst-dir> [options]

Options:
    -h --help                   Show this screen.
    -o DIRECTORY                Specify output directory [default: output].
    --compression COMP_FILTER   Specify compression filter [default: lzf].
    --cxi-size SIZE             Specify max frame in a cxi file [default: 1000].
    --cxi-dataset DATASET       Specify cxi dataset [default: data].
    --shuffle SHUFFLE           Whether use shuffle filter in compression [default: True]. 
    --batch-size SIZE           Specify batch size in a job [default: 50].
    --buffer-size SIZE          Specify buffer size in MPI communication
                                [default: 100000].
    --update-freq FREQ          Specify update frequency of progress [default: 10].
"""
from mpi4py import MPI
import numpy as np
import pandas as pd
import h5py
import time

import sys
import os
import subprocess
from glob import glob
from docopt import docopt
import yaml


def collect_jobs(files, dataset, batch_size):
    jobs = []
    batch = []
    total_h5 = 0
    for f in files:
        batch.append(f)
        total_h5 += 1
        if len(batch) == batch_size:
            jobs.append(batch)
            batch = []
    if len(batch) > 0:
        jobs.append(batch)
    return jobs, total_h5


def master_run(args):
    # mkdir for output
    cxi_dir = args['<cxi-dir>']
    cxi_lst_dir = args['<cxi-lst-dir>']
    if not os.path.isdir(cxi_dir):
        os.makedirs(cxi_dir)
    if not os.path.isdir(cxi_lst_dir):
        os.makedirs(cxi_lst_dir)
    # get all h5 files
    h5_lst = args['<h5-lst>']
    with open(h5_lst) as f:
        _files = f.readlines()
    # remove trailing '/n'
    files = []
    h5_raw_size = 0
    for f in _files:
        files.append(f[:-1])
        h5_raw_size += os.path.getsize(f[:-1])
    # collect jobs
    h5_dataset = args['<h5-dataset>']
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    jobs, total_frame = collect_jobs(files, h5_dataset, batch_size)
    total_jobs = len(jobs)
    time_start = time.time()
    print('%d frames, %d jobs to be processed' % (total_frame, total_jobs))

    update_freq = int(args['--update-freq'])

    # distribute jobs
    job_id = 0
    reqs = {}
    results = []
    slaves = list(range(1, size))
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
        if job_id % update_freq == 0:
            progress_file = os.path.join(cxi_dir, 'progress.txt')
            progress = float(job_id) / total_jobs * 100
            with open(progress_file, 'w') as f:
                f.write(str(progress))

    all_accepted = False
    while not all_accepted:
        all_accepted = True
        time.sleep(0.1)  # take a break
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                stop = True
                comm.isend(stop, dest=slave)
            else:
                all_accepted = False
    print('all jobs accepted')
    sys.stdout.flush()

    all_done = False
    for slave in slaves:
        reqs[slave] = comm.irecv(source=slave)
    while not all_done:
        all_done = True
        time.sleep(0.1)
        for slave in slaves:
            done, _ = reqs[slave].test()
            if done:
                print('slave %d done' % slave)
                sys.stdout.flush()
                slaves.remove(slave)
            else:
                all_done = False
    # save results
    time_end = time.time()
    duration = time_end - time_start

    # write cxi lst file
    cxi_raw_size = 0
    cxi_prefix = os.path.basename(h5_lst).split('.')[0]
    cxi_lst_file = open(os.path.join(cxi_lst_dir, '%s.lst' % cxi_prefix), 'w')
    cxi_files = glob('%s/*.cxi' % cxi_dir)
    for cxi_file in cxi_files:
        cxi_file = os.path.abspath(cxi_file)
        cxi_raw_size += os.path.getsize(cxi_file)
        cxi_lst_file.write('%s\n' % cxi_file)
    cxi_lst_file.close()

    compression_ratio = float(h5_raw_size) / float(cxi_raw_size)
    stat_dict = {
        'duration/sec': duration,
        'total frames': total_frame,
        'total jobs': total_jobs,
        'h5 raw size': h5_raw_size,
        'cxi raw size': cxi_raw_size,
        'compression ratio': compression_ratio,
    }
    
    stat_file = os.path.join(cxi_dir, 'stat.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    progress_file = os.path.join(cxi_dir, 'progress.txt')
    with open(progress_file, 'w') as f:
        f.write('done')
    print('All Done!')


def slave_run(args):
    stop = False
    filepath = None
    h5_obj = None
    h5_lst = args['<h5-lst>']
    h5_dataset = args['<h5-dataset>']
    cxi_dataset = args['--cxi-dataset']
    cxi_dir = args['<cxi-dir>']
    cxi_size = int(args['--cxi-size'])
    buffer_size = int(args['--buffer-size'])
    cxi_prefix = os.path.basename(h5_lst).split('.')[0]
    if args['--shuffle'] == 'True':
        shuffle = True
    else:
        shuffle = False
    done = False

    # perform h52cxi conversion
    cxi_batch = []
    cxi_count = 0
    while not stop:
        job = comm.recv(buf=buffer_size, source=0)
        for i in range(len(job)):
            cxi_batch.append(job[i])
            if len(cxi_batch) == cxi_size:
                output = os.path.join(
                    cxi_dir, 
                    '%s-rank%d-job%d.cxi' % (cxi_prefix, rank, cxi_count)
                )
                save_cxi(cxi_batch, h5_dataset, cxi_dataset, output, shuffle=shuffle)
                sys.stdout.flush()
                cxi_batch = []
                cxi_count += 1
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # save last cxi batch
    if len(cxi_batch) > 0:
        output = os.path.join(
            cxi_dir, 
            '%s-rank%d-job%d.cxi' % (cxi_prefix, rank, cxi_count)
        )
        save_cxi(cxi_batch, h5_dataset, cxi_dataset, output, shuffle=shuffle)
        sys.stdout.flush()
    done = True
    comm.send(done, dest=0)


def save_cxi(h5_files, 
             h5_dataset,
             cxi_dataset,
             cxi_file, 
             save_h5name=True,
             compression='lzf',
             shuffle=True):
    data = []
    for h5_file in h5_files:
        try:
            data.append(h5py.File(h5_file, 'r')[h5_dataset].value)
        except IOError:
            print('Warning: failed load dataset from %s' % h5_file)
            sys.stdout.flush()
            continue
    data = np.array(data).astype(np.uint16)
    n, x, y = data.shape
    f = h5py.File(cxi_file)
    f.create_dataset(
        cxi_dataset,
        shape=(n, x, y),
        dtype=np.uint16,
        data=data,
        compression=compression,
        chunks=(1, x, y),
        shuffle=True,
    )



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch h52cxi with at leat 2 processs!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        slave_run(args)
