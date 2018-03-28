import os
import sys

import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter, convolve1d
from skimage.morphology import disk


def read_image(filepath, frame=0, h5_obj=None, dataset=None):
    ext = filepath.split('.')[-1]
    if ext == 'npy':
        data = np.load(filepath)
    elif ext == 'npz':
        data = np.load(filepath)[dataset]
        if len(data.shape) == 3:
            data = data[frame]
    elif ext in ('h5', 'cxi'):
        if 'header/frame_num' in h5_obj.keys():  # PAL specific h5 file
            data = h5_obj['ts-%07d/data' % frame].value
        elif len(h5_obj[dataset].shape) == 3:
            data = h5_obj[dataset][frame]
        else:
            data = h5_obj[dataset].value
    else:
        print('Unsupported format: %s' % ext)
        return None
    return data


def find_peaks(image,
               mask=None,
               gaussian_sigma=1.,
               min_gradient=0.,
               min_distance=0,
               max_peaks=500,
               min_snr=0.,
               ):
    peaks_dict = {
        'raw': None,
        'valid': None,
        'opt': None,
        'strong': None,
    }
    raw_image = image.copy()
    if gaussian_sigma >= 0:
        image = gaussian_filter(image.astype(np.float32), gaussian_sigma)
    grad = np.gradient(image.astype(np.float32))
    grad_mag = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
    raw_peaks = peak_local_max(grad_mag,
                               exclude_border=5,
                               min_distance=int(round((min_distance - 1.) / 2.)),
                               threshold_abs=min_gradient, num_peaks=max_peaks)
    raw_peaks = np.reshape(raw_peaks, (-1, 2))
    peaks_dict['raw'] = raw_peaks
    if len(raw_peaks) == 0:
        return peaks_dict
    # mask out invalid peaks
    if mask is not None:
        valid_peak_ids = []
        for i in range(raw_peaks.shape[0]):
            peak = np.round(raw_peaks[i].astype(np.int))
            if mask[peak[0], peak[1]] == 1:
                valid_peak_ids.append(i)
        valid_peaks = raw_peaks[valid_peak_ids]
    else:
        valid_peaks = raw_peaks.copy()
    valid_peaks = np.reshape(valid_peaks, (-1, 2))
    peaks_dict['valid'] = valid_peaks
    if len(valid_peaks) == 0:
        return peaks_dict
    # refine peak location
    opt_peaks = refine_peaks(raw_image, valid_peaks)
    peaks_dict['opt'] = opt_peaks
    # remove weak peak
    snr = calc_snr(raw_image, opt_peaks)
    strong_peaks = opt_peaks[snr >= min_snr]
    peaks_dict['strong'] = strong_peaks
    return peaks_dict


def refine_peaks(image, peaks):
    opt_peaks = peaks.copy().astype(np.float32)
    crops = []
    crop_1ds = []
    for i in range(peaks.shape[0]):
        x, y = peaks[i]
        crop = image[x - 4:x + 5, y - 4:y + 5].astype(np.float32)
        crops.append(crop)
        crop_1ds.append(np.sort(crop.flatten()))
    crops = np.array(crops)
    crop_1ds = np.array(crop_1ds)
    crop_1ds_smooth = convolve1d(crop_1ds, np.ones(3), mode='nearest')
    if crop_1ds_smooth.shape[0] == 1:
        grad = np.gradient(crop_1ds_smooth[0])
        grad = np.reshape(grad, (-1, grad.size))
    else:
        grad = np.gradient(crop_1ds_smooth)[1]
    max_ids = np.argmax(grad, axis=1)
    signal_masks = []
    for i in range(len(max_ids)):
        signal_mask = (crops[i] >= crop_1ds[i, max_ids[i]]).astype(np.int)
        signal_masks.append(signal_mask)
    signal_masks = np.array(signal_masks)
    ids = (np.indices((9, 9)) - 4).astype(np.float)
    weight = np.sum(np.sum(crops * signal_masks, axis=1), axis=1)
    weight = weight.astype(np.float32)
    opt_peaks[:, 0] += np.sum(np.sum(
        crops * signal_masks * ids[0], axis=1), axis=1) / weight
    opt_peaks[:, 1] += np.sum(np.sum(
        crops * signal_masks * ids[1], axis=1), axis=1) / weight
    return opt_peaks


def calc_snr(image,
             pos,
             signal_radius=1,
             noise_inner_radius=2,
             noise_outer_radius=3,
             mode='static'):
    """
    Calculate SNR for each pos in the given image.
    """
    pos = np.round(pos).astype(np.int)
    nb_pos = len(pos)
    r1 = signal_radius
    r2, r3 = noise_inner_radius, noise_outer_radius
    crop_size = r3 * 2 + 1
    # collect crops
    crops = []
    for i in range(nb_pos):
        x, y = pos[i]
        crop = image[x-r3:x+r3+1, y-r3:y+r3+1]
        crops.append(crop)
    crops = np.array(crops).astype(np.float32)
    crops = np.reshape(crops, (-1, crop_size, crop_size))

    if mode == 'static':
        d1, d2, d3 = disk(r1), disk(r2), disk(r3)
        pad_signal, pad_BG = r3 - r1, r3 - r2
        region_signal = np.pad(d1, pad_signal, 'constant', constant_values=0)
        region_signal = np.reshape(region_signal, (1, crop_size, crop_size))
        region_signal = np.repeat(region_signal, nb_pos, axis=0)
        region_BG = (1 - np.pad(
            d2, pad_BG, 'constant', constant_values=0)) * d3
        region_BG = np.reshape(region_BG, (1, crop_size, crop_size))
        region_BG = np.repeat(region_BG, nb_pos, axis=0)
        val_BG = np.sum(np.sum(crops * region_BG, axis=1), axis=1)
        val_BG /= (np.sum(region_BG) / nb_pos)
        val_signal = np.sum(np.sum(crops * region_signal, axis=1), axis=1) 
        val_signal /= (np.sum(region_signal) / nb_pos)
        val_signal -= val_BG
        val_noise = np.std(crops[region_BG == 1].reshape((nb_pos, -1)), axis=1)
        snr = val_signal / val_noise
    return snr


def get_data_shape(filepath):
    data_shape = {}
    ext = filepath.split('.')[-1]
    if ext in ('h5', 'cxi'):
        f = h5py.File(filepath, 'r')
        keys = []

        def _get_all_dataset(key):
            if isinstance(f[key], h5py._hl.dataset.Dataset):
                keys.append(key)
        if 'header/frame_num' in f.keys():  # PAL specific h5 file
            nb_frame = f['header/frame_num'].value
            data_shape = {}
            x, y = f['ts-0000000/data'].shape
            data_shape['ts-data-PAL'] = (nb_frame, x, y)
            return data_shape
        f.visit(_get_all_dataset)
        for key in keys:
            if len(f[key].shape) in (2, 3):
                data_shape[key] = f[key].shape
        f.close()
    elif ext == 'npz':
        data = np.load(filepath)
        keys = data.keys()
        for key in keys:
            if len(data[key].shape) == 2:
                x, y = data[key].shape
                data_shape[key] = (1, x, y)
            elif len(data[key].shape) == 3:
                data_shape[key] = data[key].shape
    else:
        print('Unsupported file type: %s' % filepath)
        return
    return data_shape


def save_cxi(batch,
             out_file,
             out_dataset,
             out_dtype=None,
             compression='lzf',
             shuffle=True):
    data = []
    h5_obj = None
    filepath_curr = None
    for record in batch:
        filepath = record['filepath']
        if filepath != filepath_curr:
            try:
                h5_obj = h5py.File(filepath, 'r')
                filepath_curr = filepath
            except IOError:
                print('Failed to load %s' % filepath)
        dataset = h5_obj[record['dataset']]
        if len(dataset.shape) == 3:
            frame = dataset[record['frame']]
        elif len(dataset.shape) == 2:
            frame = dataset.value
        data.append(frame)
    in_dtype = frame.dtype
    if out_dtype == 'auto':
        out_dtype = in_dtype
    else:
        out_dtype = np.dtype(out_dtype)
    data = np.array(data).astype(out_dtype)
    n, x, y = data.shape
    if os.path.exists(out_file):
        os.rename(out_file, '%s.bk' % out_file)
    f = h5py.File(out_file, 'w')
    f.create_dataset(
        out_dataset,
        shape=(n, x, y),
        dtype=out_dtype,
        data=data,
        compression=compression,
        chunks=(1, x, y),
        shuffle=shuffle,
    )


def multiply_masks(mask_files):
    mask = 1
    for mask_file in mask_files:
        mask = np.load(mask_file) * mask
    return mask.astype(int)


def csv2cxi(csv_file, output_dir, dataset, dtype=np.int32, min_peak=20, cxi_size=100):
    df = pd.read_csv(csv_file)
    df = df[df['nb_peak'] >= min_peak]
    from_files = pd.unique(df['filepath'])
    data = []
    cxi_count = 0
    nb_cxi = int(np.ceil(len(df) / cxi_size))
    prefix = os.path.basename(csv_file).split('.')[0]
    for f in from_files:
        try:
            h5_obj = h5py.File(f, 'r')
        except IOError:
            print('Failed to load %s' % f)
            continue
        frames = df[df['filepath'] == f]['frame']
        for frame in frames:
            data.append(h5_obj[dataset][frame])
            if len(data) == cxi_size:
                output = os.path.join(output_dir, '%s-%d.cxi' % (prefix, cxi_count))
                data = np.array(data).astype(dtype)
                n, x, y = data.shape
                print('save cxi %d/%d to %s' % (cxi_count, nb_cxi, output))
                cxi_obj = h5py.File(output, 'w')
                cxi_obj.create_dataset(
                    'data',
                    shape=data.shape,
                    dtype=dtype,
                    data=data,
                    compression='lzf',
                    chunks=(1, x, y),
                    shuffle=True,
                )
                data = []
                cxi_count += 1
    if len(data) > 0:
        data = np.array(data).astype(dtype)
        output = os.path.join(output_dir, '%s-%d.cxi' % (prefix, cxi_count))
        cxi_obj = h5py.File(output, 'w')
        cxi_obj.create_dataset(
            'data',
            shape=data.shape,
            dtype=dtype,
            data=data,
            compression='lzf',
            chunks=(1, x, y),
            shuffle=True,
        )
        print('save cxi %d/%d to %s' % (cxi_count, nb_cxi, output))


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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
                    {'filepath': files[i], 'dataset': dataset, 'frame': 0}
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


def save_full_cxi(batch, cxi_file, conf=None, cxi_dtype=None, compression='lzf', shuffle=True):
    print('saving %s' % cxi_file)
    filepath_curr = None
    h5_obj = None
    data = []
    if conf['mask file'] is not None:
        mask = np.load(conf['mask file'])
    gaussian_sigma = float(conf['gaussian filter sigma'])
    max_peaks = int(conf['max peak num'])
    min_distance = int(conf['min distance'])
    min_gradient = float(conf['min gradient'])
    min_peaks = int(conf['min peak num'])
    min_snr = float(conf['min snr'])

    for i in range(len(batch)):
        record = batch[i]
        filepath = record['filepath']
        if filepath != filepath_curr:
            try:
                h5_obj = h5py.File(filepath, 'r')
                filepath_curr = filepath
            except IOError:
                print('Failed to load %s' % filepath)
                continue
        dataset = h5_obj[record['dataset']]
        if len(dataset.shape) == 3:
            frame = dataset[record['frame']]
        elif len(dataset.shape) == 2:
            frame = dataset.value
        else:
            pass
        data.append(frame)
        peaks_dict = find_peaks(
            frame, mask=mask,
            gaussian_sigma=gaussian_sigma,
            min_gradient=min_gradient,
            min_distance=min_distance,
            max_peaks=max_peaks,
            min_snr=min_snr)
        strong_peaks = peaks_dict['strong']
    # in_dtype = frame.dtype
    # if cxi_dtype == 'auto':
    #     cxi_dtype = in_dtype
    # else:
    #     cxi_dtype = np.dtype(cxi_dtype)
    # data = np.array(data).astype(cxi_dtype)
    # n, x, y = data.shape
    # if os.path.exists(cxi_file):
    #     os.rename(cxi_file, '%s.bk' % cxi_file)
    # f = h5py.File(cxi_file, 'w')
    # # patterns
    # f.create_dataset(
    #     'data',
    #     shape=(n, x, y),
    #     dtype=cxi_dtype,
    #     data=data,
    #     compression=compression,
    #     chunks=(1, x, y),
    #     shuffle=shuffle,
    # )
    #
    # # peaks
