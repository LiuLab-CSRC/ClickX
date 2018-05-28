#!/bin/env python

"""
Compress raw data in h5/cxi format to compressed cxi files on multiple cores
using MPI.

Usage:
   batch_compressor.py <raw-lst> <raw-dataset> <comp-dir> <comp-lst-dir> [options]

Options:
    -h --help                   Show this screen.
    --compression COMP_FILTER   Specify compression filter [default: lzf].
    --comp-size SIZE            Specify max frame in a compressed cxi file
                                [default: 1000].
    --comp-dataset DATASET      Specify dataset for patterns in compressed cxi
                                file [default: data].
    --comp-dtype DATATYPE       Specify datatype of patterns in compressed cxi
                                file [default: auto].
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
import sys
import os
import time
from glob import glob
from docopt import docopt
import yaml

from mpi4py import MPI
import util


def master_run(args):
    flush = args['--flush']
    # mkdir for output if not exist
    comp_dir = os.path.abspath(args['<comp-dir>'])
    comp_lst_dir = os.path.abspath(args['<comp-lst-dir>'])
    if not os.path.isdir(comp_dir):
        os.makedirs(comp_dir)
    if not os.path.isdir(comp_lst_dir):
        os.makedirs(comp_lst_dir)
    # get all raw data files
    raw_lst = args['<raw-lst>']
    with open(raw_lst) as f:
        _files = f.readlines()
    # remove trailing '/n'
    files = []
    raw_size = 0  # total size fo input files(raw data)
    for f in _files:
        files.append(f[:-1])
        raw_size += os.path.getsize(f[:-1])
    # collect jobs
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    raw_dataset = args['<raw-dataset>']
    jobs, nb_frames = util.collect_jobs(files, raw_dataset, batch_size)
    nb_jobs = len(jobs)
    time_start = time.time()
    print('%d frames, %d jobs to be processed' %
          (nb_frames, nb_jobs), flush=flush)

    update_freq = int(args['--update-freq'])

    # distribute jobs
    job_id = 0
    reqs = {}
    results = []
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(job, dest=worker)
        reqs[worker] = comm.irecv(buf=buffer_size, source=worker)
        print('job %d/%d --> %d' % (job_id, nb_jobs, worker), flush=flush)
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        time.sleep(0.1)  # take a break
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                results += result
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
                    finished_workers.add(worker)
        if job_id % update_freq == 0:
            progress = float(job_id) / nb_jobs * 100
            stat_dict = {
                'time start': time_start,
                'progress': '%.2f%%' % progress,
                'duration/sec': 'not finished',
                'total frames': nb_frames,
                'total jobs': nb_jobs,
                'raw size': 'not finished',
                'compressed size': 'not finished',
                'compression ratio': 'not finished',
            }

            stat_file = os.path.join(comp_dir, 'stat.yml')
            with open(stat_file, 'w') as f:
                yaml.dump(stat_dict, f, default_flow_style=False)

    all_accepted = False
    while not all_accepted:
        all_accepted = True
        time.sleep(0.1)  # take a break
        workers -= finished_workers
        for worker in workers:
            finished, result = reqs[worker].test()
            if finished:
                stop = True
                comm.isend(stop, dest=worker)
                finished_workers.add(worker)
            else:
                all_accepted = False
    print('all jobs accepted', flush=flush)

    # check status of last job batch
    all_done = False
    workers = set(range(1, size))
    finished_workers = set()
    for worker in workers:
        reqs[worker] = comm.irecv(source=worker)
    while not all_done:
        all_done = True
        time.sleep(0.1)
        workers -= finished_workers
        for worker in workers:
            done, _ = reqs[worker].test()
            if done:
                print('worker %d done' % worker, flush=flush)
                finished_workers.add(worker)
            else:
                all_done = False
    # save results
    time_end = time.time()
    duration = time_end - time_start

    # write compressed cxi file lst
    compressed_size = 0
    prefix = os.path.basename(raw_lst).split('.')[0]
    comp_lst_file = open(os.path.join(comp_lst_dir, '%s.lst' % prefix), 'w')
    comp_files = glob('%s/*.cxi' % comp_dir)
    for comp_file in comp_files:
        compressed_size += os.path.getsize(os.path.abspath(comp_file))
        comp_lst_file.write('%s\n' % comp_file)
    comp_lst_file.close()

    compression_ratio = float(raw_size) / float(compressed_size)
    stat_dict = {
        'time start': time_start,
        'progress': 'done',
        'duration/sec': duration,
        'total frames': nb_frames,
        'total jobs': nb_jobs,
        'raw size': raw_size,
        'compressed size': compressed_size,
        'compression ratio': '%.2f' % compression_ratio,
    }
    
    stat_file = os.path.join(comp_dir, 'stat.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    print('All Done!', flush=flush)


def worker_run(args):
    stop = False
    flush = args['--flush']
    in_lst = args['<raw-lst>']
    comp_dir = args['<comp-dir>']
    comp_dataset = args['--comp-dataset']
    comp_size = int(args['--comp-size'])
    buffer_size = int(args['--buffer-size'])
    prefix = os.path.basename(in_lst).split('.')[0]
    comp_dtype = args['--comp-dtype']
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
            batch.append(job[i])
            if len(batch) == comp_size:
                comp_file = os.path.join(
                    comp_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
                )
                util.save_cxi(
                    batch, comp_file, comp_dataset,
                    out_dtype=comp_dtype, shuffle=shuffle
                )
                batch = []
                count += 1
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # save last cxi batch if not empty
    if len(batch) > 0:
        comp_file = os.path.join(
            comp_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
        )
        util.save_cxi(
            batch, comp_file, comp_dataset,
            out_dtype=comp_dtype, shuffle=shuffle
        )
    done = True
    comm.send(done, dest=0)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch compressor with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        worker_run(args)
