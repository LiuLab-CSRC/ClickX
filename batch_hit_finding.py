"""
Run hit finding on multiple cores using MPI.
Input:
    multiple cxi files
    hit finding conf file
Output:
    hits file in cxi format
"""
from mpi4py import MPI
import numpy as np
import time
import sys
from glob import glob
import h5py
import yaml

from util import find_peaks, read_image
import pandas as pd


BATCH_SIZE = 5
BUFFER_SIZE = 100000  # incease buffersize as needed


def batch_hit_finding(jobs, conf):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    master = 0
    nb_jobs = len(jobs)

    dataset = conf['dataset']

    # MASTER
    if rank == master:
        job_id = 0
        nb_slaves = size - 1
        reqs = {}
        results = []
        for i in range(nb_slaves):
            slave = i+1
            comm.isend(jobs[job_id], dest=slave)
            reqs[slave] = comm.irecv(buf=BUFFER_SIZE, source=slave)
            print('send job %d/%d to %d' % (job_id, nb_jobs, slave))
            sys.stdout.flush()
            job_id += 1
        while job_id < nb_jobs:
            stop = False
            for i in range(nb_slaves):
                slave = i + 1
                finished, result = reqs[slave].test()
                if finished:
                    print('slave %d finished %s' % (slave, str(result)))
                    results += result
                    if job_id < nb_jobs:
                        print('send job %d/%d to %d' % (
                            job_id, nb_jobs, slave))
                        sys.stdout.flush()
                        comm.isend(stop, dest=slave)
                        comm.isend(jobs[job_id], dest=slave)
                        reqs[slave] = comm.irecv(buf=BUFFER_SIZE, source=slave)
                        job_id += 1
                    else:
                        stop = True
                        comm.isend(stop, dest=slave)
                        print('send stop signal to %d' % slave)
        print('[MASTER] Sent all jobs')

        all_done = False
        while not all_done:
            all_done = True
            for i in range(nb_slaves):
                slave = i + 1
                if reqs[slave].test():
                    stop = True
                    comm.isend(stop, dest=slave)
                else:
                    all_done = False
        # save results
        df = pd.DataFrame(results)
        df.to_csv('results.csv')
        print('Done!')

    # SLAVES
    else:
        stop = False
        filepath = None
        h5_obj = None

        # hit finding parameters
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
        refine_on = conf['refine on']

        while not stop:
            job = comm.recv(buf=BUFFER_SIZE, source=master)
            # perform job
            for i in range(len(job)):
                _filepath = job[i]['filepath']
                frame = job[i]['frame']
                if _filepath != filepath:
                    filepath = _filepath
                    h5_obj = h5py.File(filepath, 'r')
                image = read_image(filepath, frame=frame,
                                   h5_obj=h5_obj, h5_dataset=dataset)
                peaks = find_peaks(image, mask=mask,
                                   gaussian_sigma=gaussian_sigma,
                                   min_distance=min_distance,
                                   min_gradient=min_gradient,
                                   max_peaks=max_pean_num,
                                   min_snr=min_snr,
                                   refine=refine_on,
                                   )
                job[i]['nb_peak'] = len(peaks)
                # print(job[i])
                # print(len(peaks))
                # sys.stdout.flush()
            comm.send(job, dest=master)
            stop = comm.recv(source=master)
        print('slave %d is exiting' % rank)


def collect_jobs(files, dataset):
    jobs = []
    batch = []
    for f in files:
        try:
            shape = h5py.File(f, 'r')[dataset].shape
            if len(shape) == 3:
                nb_frame = shape[0]
                for i in range(nb_frame):
                    batch.append({'filepath': f, 'frame': i})
            else:
                batch.append({'filepath': f, 'frame': 0})
            if len(batch) == BATCH_SIZE:
                jobs.append(batch)
                batch = []
        except OSError:
            pass
    if len(batch) > 0:
        jobs.append(batch)
    return jobs


if __name__ == '__main__':
    files = glob('/Volumes/LaCie/data/temp/data1/*.h5')
    conf_file = '/Volumes/LaCie/data/temp/conf.yml'
    with open(conf_file, 'r') as f:
        conf = yaml.load(f)
    jobs = collect_jobs(files, conf['dataset'])
    batch_hit_finding(jobs[:10], conf)
