#!/bin/env python

"""Generate cxi files from hits in csv files using MPI.

Usage:
   batch_csv2cxi.py <csv-file> <conf-file> <hit-dir> [options]

Options:
    -h --help                   Show this screen.
    --cxi-size SIZE             Specify max frame in a cxi file [default: 1000].
    --cxi-dtype DATATYPE        Specify datatype of patterns in compressed cxi file [default: auto].
    --shuffle SHUFFLE           Whether to use shuffle filter in compression [default: True].
    --batch-size SIZE           Specify batch size in a job [default: 50].
    --buffer-size SIZE          Specify buffer size in MPI communication
                                [default: 100000].
    --update-freq FREQ          Specify update frequency of progress [default: 10].

"""
from mpi4py import MPI
from docopt import docopt
import sys
import os
import time
import pandas as pd
import numpy as np
import yaml
from util import save_full_cxi


def master_run(args):
    # mkdir if not exist
    hit_dir = args['<hit-dir>']
    if not os.path.isdir(hit_dir):
        os.mkdir(hit_dir)
    csv_file = args['<csv-file>']

    conf_file = args['<conf-file>']
    with open(conf_file, 'r') as f:
        conf = yaml.load(f)
    min_peaks = conf['min peak num']

    df = pd.read_csv(csv_file)
    df = df[df['nb_peak'] > min_peaks]

    batch_size = int(args['--batch-size'])
    nb_jobs = int(np.ceil(len(df) / batch_size))

    # collect jobs
    ids = np.array_split(np.arange(len(df)), nb_jobs)
    jobs = []
    for i in range(len(ids)):
        jobs.append(df.iloc[ids[i]])
    print('%d jobs, %d frames to be processed' % (nb_jobs, len(df)))

    # other parameters
    buffer_size = int(args['--buffer-size'])
    update_freq = int(args['--update-freq'])

    # distribute jobs
    job_id = 0
    reqs = {}
    slaves = list(range(1, size))
    time_start = time.time()
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
                    print('stop signal sent to %d' % slave)
        if job_id % update_freq == 0:
            # update stat
            progress = float(job_id) / nb_jobs * 100
            stat_dict = {
                'progress': '%.2f%%' % progress,
                'duration/sec': 'not finished',
                'total jobs': nb_jobs,
                'time start': time_start,
            }
            stat_file = os.path.join(hit_dir, 'csv2cxi.yml')
            with open(stat_file, 'w') as f:
                yaml.dump(stat_dict, f, default_flow_style=False)

    all_done = False
    while not all_done:
        all_done = True
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                slaves.remove(slave)
                stop = True
                comm.isend(stop, dest=slave)
            else:
                all_done = False
    time_end = time.time()
    duration = time_end - time_start

    stat_dict = {
        'progress': 'done',
        'duration/sec': duration,
        'total jobs': nb_jobs,
        'time start': time_start,
    }
    stat_file = os.path.join(hit_dir, 'csv2cxi.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    print('All Done!')


def slave_run(args):
    stop = False
    csv_file = args['<csv-file>']
    hit_dir = args['<hit-dir>']
    conf_file = args['<conf-file>']
    with open(conf_file, 'r') as f:
        conf = yaml.load(f)
    cxi_size = int(args['--cxi-size'])
    cxi_dtype = args['--cxi-dtype']
    buffer_size = int(args['--buffer-size'])
    prefix = os.path.basename(csv_file).split('.')[0]
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
            batch.append(job.iloc[i])
            if len(batch) == cxi_size:
                # save full cxi file
                cxi_file = os.path.join(
                    hit_dir, '%s-rank%d-job%d.cxi' % (prefix, rank, count)
                )
                save_full_cxi(batch, cxi_file, conf=conf, cxi_dtype=cxi_dtype, shuffle=shuffle)
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
        save_full_cxi(batch, cxi_file, cxi_dtype=cxi_dtype, shuffle=shuffle)
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
