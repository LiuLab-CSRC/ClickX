#!/bin/env python

"""Compress raw data in h5/cxi format to compressed cxi files on multiple cores using MPI.

Usage:
   batch_compressor.py <in-lst> <in-dataset> <out-dir> <out-lst-dir> [options]

Options:
    -h --help                   Show this screen.
    --compression COMP_FILTER   Specify compression filter [default: lzf].
    --out-size SIZE             Specify max frame in a cxi file [default: 1000].
    --out-dataset DATASET       Specify cxi dataset [default: data].
    --out-dtype DATATYPE        Specify the datatype [default: auto].
    --shuffle SHUFFLE           Whether use shuffle filter in compression [default: True]. 
    --batch-size SIZE           Specify batch size in a job [default: 50].
    --buffer-size SIZE          Specify buffer size in MPI communication
                                [default: 100000].
    --update-freq FREQ          Specify update frequency of progress [default: 10].
"""
from mpi4py import MPI
import h5py

from util import save_cxi
import time

import sys
import os
from glob import glob
from docopt import docopt
import yaml
from tqdm import tqdm


def collect_jobs(files, dataset, batch_size):
    jobs = []
    batch = []
    frames = 0
    print('collecting jobs...')
    for i in tqdm(range(len(files))):
        try:
            shape = h5py.File(files[i], 'r')[dataset].shape
            if len(shape) == 3:
                nb_frame = shape[0]
                for j in range(nb_frame):
                    batch.append(
                        {'filepath': files[i], 'dataset': dataset, 'frame': j}
                    )
                    frames += 1
                    if len(batch) == batch_size:
                        jobs.append(batch)
                        batch = []
            else:
                batch.append(
                    {'filepath': files[i], 'dataset': dataset, 'frame': j}
                )
                frames += 1
                if len(batch) == batch_size:
                    jobs.append(batch)
                    batch = []
        except OSError:
            print('Failed to load %s' % files[i])
            pass
    if len(batch) > 0:
        jobs.append(batch)
    return jobs, frames


def master_run(args):
    # mkdir for output if not exist
    out_dir = os.path.abspath(args['<out-dir>'])
    out_lst_dir = os.path.abspath(args['<out-lst-dir>'])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(out_lst_dir):
        os.makedirs(out_lst_dir)
    # get all h5 files
    in_lst = args['<in-lst>']
    with open(in_lst) as f:
        _files = f.readlines()
    # remove trailing '/n'
    files = []
    in_size = 0  # total size fo input files(raw data)
    for f in _files:
        files.append(f[:-1])
        in_size += os.path.getsize(f[:-1])
    # collect jobs
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    in_dataset = args['<in-dataset>']
    jobs, nb_frames = collect_jobs(files, in_dataset, batch_size)
    nb_jobs = len(jobs)
    time_start = time.time()
    print('%d frames, %d jobs to be processed' % (nb_frames, nb_jobs))

    update_freq = int(args['--update-freq'])

    # distribute jobs
    job_id = 0
    reqs = {}
    results = []
    slaves = list(range(1, size))
    for slave in slaves:
        comm.isend(jobs[job_id], dest=slave)
        reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
        print('job %d/%d sent to %d' % (job_id, nb_jobs, slave))
        sys.stdout.flush()
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        time.sleep(0.1)  # take a break
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                results += result
                if job_id < nb_jobs:
                    print('job %d/%d sent to %d' % (job_id, nb_jobs, slave))
                    sys.stdout.flush()
                    comm.isend(stop, dest=slave)
                    comm.isend(jobs[job_id], dest=slave)
                    reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=slave)
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

            stat_file = os.path.join(out_dir, 'stat.yml')
            with open(stat_file, 'w') as f:
                yaml.dump(stat_dict, f, default_flow_style=False)

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

    # write compressed cxi file lst
    compressed_size = 0
    prefix = os.path.basename(in_lst).split('.')[0]
    out_lst_file = open(os.path.join(out_lst_dir, '%s.lst' % prefix), 'w')
    out_files = glob('%s/*.cxi' % out_dir)
    for out_file in out_files:
        compressed_size += os.path.getsize(os.path.abspath(out_file))
        out_lst_file.write('%s\n' % out_file)
    out_lst_file.close()

    compression_ratio = float(in_size) / float(compressed_size)
    stat_dict = {
        'time start': time_start,
        'progress': 'done',
        'duration/sec': duration,
        'total frames': nb_frames,
        'total jobs': nb_jobs,
        'raw size': in_size,
        'compressed size': compressed_size,
        'compression ratio': '%.2f' % compression_ratio,
    }
    
    stat_file = os.path.join(out_dir, 'stat.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    print('All Done!')


def slave_run(args):
    stop = False
    in_lst = args['<in-lst>']
    out_dir = args['<out-dir>']
    out_dataset = args['--out-dataset']
    out_size = int(args['--out-size'])
    buffer_size = int(args['--buffer-size'])
    prefix = os.path.basename(in_lst).split('.')[0]
    out_dtype = args['--out-dtype']
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
            if len(batch) == out_size:
                out_file = os.path.join(
                    out_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
                )
                save_cxi(batch, out_file, out_dataset, out_dtype=out_dtype, shuffle=shuffle)
                sys.stdout.flush()
                batch.clear()
                count += 1
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    # save last cxi batch if not empty
    if len(batch) > 0:
        out_file = os.path.join(
            out_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
        )
        save_cxi(batch, out_file, out_dataset, out_dtype=out_dtype, shuffle=shuffle)
        sys.stdout.flush()
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
        slave_run(args)
