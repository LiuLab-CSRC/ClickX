#!/bin/env python

"""Run hit finding on multiple cores using MPI.

Usage:
   batch_hit_finder.py <cxi-lst> <conf-file> <hit-dir> [options]

Options:
    -h --help               Show this screen.
    --min-peak NUM          Specify min peaks for a hit [default: 20].
    --batch-size SIZE       Specify batch size in a job [default: 10].
    --buffer-size SIZE      Specify buffer size in MPI communication
                            [default: 100000].
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

import util


def master_run(args):
    # mkdir if not exist
    hit_dir = args['<hit-dir>']
    if not os.path.isdir(hit_dir):
        os.makedirs(hit_dir)
    cxi_lst = args['<cxi-lst>']
    min_peak = int(args['--min-peak'])
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
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    jobs, nb_frames = util.collect_jobs(files, dataset, batch_size)
    nb_jobs = len(jobs)
    print('%d frames, %d jobs to be processed' % (nb_frames, nb_jobs))

    update_freq = int(args['--update-freq'])
    prefix = os.path.basename(cxi_lst).split('.')[0]

    # dispatch jobs
    job_id = 0
    reqs = {}
    results = []
    slaves = set(range(1, size))
    finished_slaves = set()
    time_start = time.time()
    for slave in slaves:
        if job_id < nb_jobs:
            job = jobs[job_id]
        else:
            job = []  # dummy job
        comm.isend(jobs[job_id], dest=slave)
        reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
        print('job %d/%d --> slave %d' % (job_id, nb_jobs, slave), flush=True)
        job_id += 1
    while job_id < nb_jobs:
        stop = False
        time.sleep(0.1)  # take a break
        slaves -= finished_slaves
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                results += result
                if job_id < nb_jobs:
                    print('job %d/%d --> slave %d' %
                          (job_id, nb_jobs, slave), flush=True)
                    comm.isend(stop, dest=slave)
                    comm.isend(jobs[job_id], dest=slave)
                    reqs[slave] = comm.irecv(buf=buffer_size, source=slave)
                    job_id += 1
                else:
                    stop = True
                    comm.isend(stop, dest=slave)
                    print('stop signal --> slave %d' % slave)
                    finished_slaves.add(slave)
        if job_id % update_freq == 0:
            # update stat
            progress = float(job_id) / nb_jobs * 100
            df = pd.DataFrame(results)
            processed_hits = len(df[df['nb_peak'] >= min_peak])
            processed_frames = len(df)
            hit_rate = float(processed_hits) / processed_frames * 100.
            stat_dict = {
                'progress': '%.2f%%' % progress,
                'processed hits': processed_hits,
                'hit rate': '%.2f%%' % hit_rate,
                'duration/sec': 'not finished',
                'processed frames': processed_frames,
                'total jobs': nb_jobs,
                'time start': time_start,
            }
            stat_file = os.path.join(hit_dir, 'stat.yml')
            with open(stat_file, 'w') as f:
                yaml.dump(stat_dict, f, default_flow_style=False)

    all_done = False
    while not all_done:
        time.sleep(0.1)
        all_done = True
        slaves -= finished_slaves
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                results += result
                stop = True
                print('stop signal --> slave %d' % slave, flush=True)
                comm.isend(stop, dest=slave)
                finished_slaves.add(slave)
            else:
                all_done = False
    time_end = time.time()
    duration = time_end - time_start
    # save stat file
    df = pd.DataFrame(results)
    processed_hits = len(df[df['nb_peak'] >= min_peak])
    processed_frames = len(df)
    hit_rate = float(processed_hits) / processed_frames * 100.
    stat_dict = {
        'progress': 'done',
        'processed hits': processed_hits,
        'hit rate': '%.2f%%' % hit_rate,
        'duration/sec': duration,
        'processed frames': processed_frames,
        'total jobs': nb_jobs,
        'time start': time_start,
    }

    stat_file = os.path.join(hit_dir, 'stat.yml')
    with open(stat_file, 'w') as f:
        yaml.dump(stat_dict, f, default_flow_style=False)

    # save simple results to csv
    csv_file = os.path.join(hit_dir, '%s.csv' % prefix)
    simple_results = []
    for i in range(len(results)):
        simple_results.append(
            {
                'filepath': results[i]['filepath'],
                'dataset': results[i]['dataset'],
                'frame': results[i]['frame'],
                'nb_peak': results[i]['nb_peak']
            }
        )
    df = pd.DataFrame(simple_results)
    df.to_csv(csv_file)

    # save detailed peak info to npz
    peak_file = os.path.join(hit_dir, '%s.npy' % prefix)
    np.save(peak_file, results)

    print('All Done!', flush=True)
    MPI.Finalize()


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
        mask = util.read_image(mask_file)
    else:
        mask = None
    max_peak_num = conf['max peak num']
    min_distance = conf['min distance']
    min_gradient = conf['min gradient']
    min_snr = conf['min snr']
    dataset = conf['dataset']
    peak_refine_mode = conf['peak refine mode']
    min_pixels = conf['min pixels']
    snr_mode = conf['snr mode']
    signal_radius = conf['signal radius']
    bg_inner_radius = conf['background inner radius']
    bg_outer_radius = conf['background outer radius']
    crop_size = conf['crop size']
    bg_ratio = conf['background ratio']
    signal_ratio = conf['signal ratio']
    signal_thres = conf['signal threshold']

    # perform hit finding
    while not stop:
        job = comm.recv(buf=buffer_size, source=0)
        for i in range(len(job)):
            _filepath = job[i]['filepath']
            frame = job[i]['frame']
            if _filepath != filepath:
                filepath = _filepath
                h5_obj = h5py.File(filepath, 'r')
            image = util.read_image(filepath,
                                    frame=frame,
                                    h5_obj=h5_obj,
                                    dataset=dataset)
            peaks_dict = util.find_peaks(
                image,
                mask=mask,
                gaussian_sigma=gaussian_sigma,
                min_distance=min_distance,
                min_gradient=min_gradient,
                max_peaks=max_peak_num,
                min_snr=min_snr,
                min_pixels=min_pixels,
                refine_mode=peak_refine_mode,
                snr_mode=snr_mode,
                signal_radius=signal_radius,
                bg_inner_radius=bg_inner_radius,
                bg_outer_radius=bg_outer_radius,
                crop_size=crop_size,
                bg_ratio=bg_ratio,
                signal_ratio=signal_ratio,
                signal_thres=signal_thres,
                label_pixels=False,
            )
            if peaks_dict['strong'] is not None:
                job[i]['nb_peak'] = len(peaks_dict['strong'])
                job[i]['peak_info'] = peaks_dict['info']
            else:
                job[i]['nb_peak'] = 0
        comm.send(job, dest=0)
        stop = comm.recv(source=0)
    print('slave %d is exiting' % rank)


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
        slave_run(argv)
