#!/bin/env python

"""
Generate peak powder pattern on multiple cores using MPI.

Usage:
   batch_peak_powder.py <file-lst> <conf-file> [options]

Options:
    -h --help               Show this screen.
    -o OUTPUT-DIR           Specify output directory [default: output].
    -p PREFIX               Specify the prefix of output files
                            [default: powder].
    --batch-size SIZE       Specify batch size in a job [default: 10].
    --buffer-size SIZE      Specify buffer size in MPI communication
                            [default: 100000].
"""
from mpi4py import MPI
import h5py
import numpy as np
import time

import sys
import os
from docopt import docopt
from ruamel.yaml import YAML

from util import find_peaks, read_image, collect_jobs


def master_run(args):
    # mkdir if not exist
    output_dir = args['-o']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    prefix = args['-p']

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
        yaml = YAML()
        conf = yaml.load(f)
    # collect jobs
    dataset = conf['dataset']
    batch_size = int(args['--batch-size'])
    buffer_size = int(args['--buffer-size'])
    jobs, nb_frames = collect_jobs(files, dataset, batch_size)
    nb_jobs = len(jobs)
    print('%d frames, %d jobs to be processed' % (nb_frames, nb_jobs))

    # dispatch jobs
    job_id = 0
    reqs = {}
    peaks = []
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
                peaks += result
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

    all_done = False
    while not all_done:
        all_done = True
        for slave in slaves:
            finished, result = reqs[slave].test()
            if finished:
                peaks += result
                slaves.remove(slave)
                stop = True
                print('send stop signal to %d' % slave)
                sys.stdout.flush()
                comm.isend(stop, dest=slave)
            else:
                all_done = False

    # save peaks
    filepath = jobs[0][0]['filepath']
    frame = jobs[0][0]['frame']
    h5_obj = h5py.File(filepath, 'r')
    image = read_image(filepath, frame=frame, h5_obj=h5_obj, dataset=dataset)
    result_dict = dict()
    result_dict['shape'] = image.shape
    result_dict['peaks'] = np.round(np.array(peaks), decimals=3).tolist()
    peak_file = os.path.join(output_dir, '%s.peaks' % prefix)
    with open(peak_file, 'w') as f:
        yaml = YAML(typ='safe')
        yaml.dump(result_dict, f)

    # build and save peak powder
    powder = np.zeros(image.shape)
    peaks = np.round(np.array(peaks)).astype(np.int)
    powder[peaks[:, 0], peaks[:, 1]] = 1
    powder_file = os.path.join(output_dir, '%s.npy' % prefix)
    np.save(powder_file, powder)
    print('All Done!')


def slave_run(args):
    stop = False
    filepath = None
    h5_obj = None
    buffer_size = int(args['--buffer-size'])

    # hit finding parameters
    with open(args['<conf-file>']) as f:
        yaml = YAML()
        conf = yaml.load(f)
    gaussian_sigma = conf['gaussian filter sigma']
    mask_file = conf['mask file']
    if mask_file is not None:
        mask = read_image(mask_file)
    else:
        mask = None
    max_peak_num = conf['max peak num']
    min_distance = conf['min distance']
    min_gradient = conf['min gradient']
    min_snr = conf['min snr']
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
            image = read_image(filepath, frame=frame,
                               h5_obj=h5_obj, dataset=dataset)
            peaks_dict = find_peaks(image, mask=mask,
                                    gaussian_sigma=gaussian_sigma,
                                    min_distance=min_distance,
                                    min_gradient=min_gradient,
                                    max_peaks=max_peak_num,
                                    min_snr=min_snr)
            if peaks_dict['strong'] is not None:
                peaks += peaks_dict['strong'].tolist()
        comm.send(peaks, dest=0)
        stop = comm.recv(source=0)
        if stop:
            print('slave %d is exiting' % rank)
            sys.stdout.flush()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print('Run batch hit finder with at least 2 processes!')
        sys.exit()

    rank = comm.Get_rank()
    args = docopt(__doc__)
    if rank == 0:
        master_run(args)
    else:
        slave_run(args)
