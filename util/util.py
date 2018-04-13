import os
import sys

import numpy as np
from scipy.linalg import eig, inv, det
from scipy.optimize import minimize
import math
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
               min_pixels=2,
               refine_mode='mean',
               snr_mode='rings',
               signal_radius=1,
               bg_inner_radius=2,
               bg_outer_radius=3,
               crop_size=7,
               bg_ratio=0.7,
               signal_ratio=0.2,
               signal_thres=5.,
               label_pixels=False):
    peaks_dict = {
        'raw': None,  # coordinates of raw peak
        'valid': None,  # coordinates of valid peak after mask out bad peak
        'opt': None,  # coordinates of optimized peak
        'strong': None,  # coordinates of strong peak with high snr
        'info': None,  # strong peak info, including intensity, snr, pixel num
    }
    raw_image = image.copy()
    if gaussian_sigma >= 0:
        image = gaussian_filter(image.astype(np.float32), gaussian_sigma)
    grad = np.gradient(image.astype(np.float32))
    grad_mag = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
    raw_peaks = peak_local_max(
        grad_mag,
        exclude_border=5,
        min_distance=int(round((min_distance - 1.) / 2.)),
        threshold_abs=min_gradient, num_peaks=max_peaks
    )
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
    opt_peaks = refine_peaks(raw_image, valid_peaks, mode=refine_mode)
    peaks_dict['opt'] = opt_peaks
    snr_info = calc_snr(
        raw_image, opt_peaks,
        mode=snr_mode,
        signal_radius=signal_radius,
        bg_inner_radius=bg_inner_radius,
        bg_outer_radius=bg_outer_radius,
        crop_size=crop_size,
        bg_ratio=bg_ratio,
        signal_ratio=signal_ratio,
        signal_thres=signal_thres,
        label_pixels=label_pixels,
    )
    strong_ids = np.where(
        (snr_info['snr'] >= min_snr) *
        (snr_info['signal pixel num'] >= min_pixels)
    )[0]
    strong_peaks = opt_peaks[strong_ids]
    peaks_dict['strong'] = strong_peaks
    peaks_dict['info'] = {
        'pos': strong_peaks,
        'snr': snr_info['snr'][strong_ids],
        'total intensity': snr_info['total intensity'][strong_ids],
        'signal values': snr_info['signal values'][strong_ids],
        'background values': snr_info['background values'][strong_ids],
        'noise values': snr_info['noise values'][strong_ids],
        'signal pixel num': snr_info['signal pixel num'][strong_ids],
        'background pixel num': snr_info['background pixel num'][strong_ids],
    }
    return peaks_dict


def refine_peaks(image, peaks, crop_size=7, mode='gradient'):
    """
    Refine peak position with adaptive method.
    :param image: 2d array.
    :param peaks: peak coordinates with shape of (N, 2)
    :param crop_size: size of crop used for refinement
    :param mode: support 'gradient' and 'mean'
    :return: optimized peak position.
    """
    image = np.array(image)
    peaks = np.round(peaks).astype(np.int).reshape(-1, 2)
    if len(image.shape) != 2:
        raise ValueError('image must be 2d array.')
    opt_peaks = peaks.copy().astype(np.float32)
    crops = []
    half_crop = crop_size // 2
    crop_1ds = []
    nb_peaks = peaks.shape[0]
    for i in range(nb_peaks):
        x, y = peaks[i]
        crop = image[x - half_crop:x + half_crop + 1,
                     y - half_crop:y + half_crop + 1]
        crops.append(crop)
        crop_1ds.append(np.sort(crop.flatten()))
    crops = np.array(crops)
    if mode == 'gradient':
        crop_1ds = np.array(crop_1ds)
        crop_1ds_smooth = convolve1d(crop_1ds, np.ones(3), mode='nearest')
        if crop_1ds_smooth.shape[0] == 1:
            grad = np.gradient(crop_1ds_smooth[0])
            grad = np.reshape(grad, (-1, grad.size))
        else:
            grad = np.gradient(crop_1ds_smooth)[1]
        max_ids = np.argmax(grad, axis=1)
        hinge = crop_1ds[np.arange(nb_peaks), max_ids].reshape(-1, 1, 1)
        hot_masks = crops >= hinge
        calib_crops = crops
    elif mode == 'mean':
        crops_mean = np.mean(np.mean(crops, axis=1), axis=1).reshape(-1, 1, 1)
        calib_crops = crops - crops_mean
        hot_masks = calib_crops > 0
    else:
        raise ValueError('only support gradient and mean mode.')
    ids = (np.indices((crop_size, crop_size)) - half_crop).astype(np.float)
    weight = np.sum(np.sum(calib_crops * hot_masks, axis=1), axis=1)
    weight = weight.astype(np.float32)
    opt_peaks[:, 0] += np.sum(np.sum(
        calib_crops * hot_masks * ids[0], axis=1), axis=1) / weight
    opt_peaks[:, 1] += np.sum(np.sum(
        calib_crops * hot_masks * ids[1], axis=1), axis=1) / weight
    return opt_peaks


def calc_snr(image,
             pos,
             mode='adaptive',
             signal_radius=1,
             bg_inner_radius=2,
             bg_outer_radius=3,
             crop_size=7,
             bg_ratio=0.7,
             signal_ratio=0.2,
             signal_thres=5.,
             label_pixels=True):
    """
    Calculate snr for given position on image.
    :param label_pixels: whether label signal/bg pixels.
    :param signal_ratio: signal pixel ratio.
    :param image: 2d array.
    :param pos: positions to calculate snr.
    :param mode: support 'simple', 'rings' and 'adaptive' modes.
    :param signal_radius: signal region radius, used in 'rings' mode.
    :param bg_inner_radius: noise inner radius, used in 'rings mode.
    :param bg_outer_radius: noise outer radius, used in 'rings mode.
    :param crop_size: size of crop for snr estimation, used in 'simple' mode.
    :param bg_ratio: background pixel ratio, used in 'simple' mode.
    :param signal_thres: float, pixel with higher value than bg + \
                         signal_thres * noise is considered as signal pixel.
    :return: snr_info dict (including snr, signal pixels, background pixels)
    """
    image = np.array(image)
    pos = np.round(np.array(pos)).astype(np.int).reshape(-1, 2)
    if len(image.shape) != 2:
        raise ValueError('image must be 2d array.')
    nb_pos = pos.shape[0]
    r1 = signal_radius
    r2, r3 = bg_inner_radius, bg_outer_radius
    # collect crops
    crops = []
    if mode == 'rings':
        half_crop = r3
    else:
        half_crop = crop_size // 2
    for i in range(nb_pos):
        x, y = pos[i]
        crop = image[x - half_crop:x + half_crop + 1,
                     y - half_crop:y + half_crop + 1]
        crops.append(crop)
    crops = np.array(crops).astype(np.float32)
    crops = np.reshape(crops, (-1, crop_size, crop_size))

    # collect statistics
    val_bg = []
    val_signal = []
    val_noise = []
    val_total_intensity = []
    nb_signal_pixels = []
    nb_bg_pixels = []
    if mode == 'rings':
        d1, d2, d3 = disk(r1), disk(r2), disk(r3)
        pad_signal, pad_bg = r3 - r1, r3 - r2
        region_signal = np.pad(
            d1, pad_signal, 'constant', constant_values=0).astype(bool)
        region_signal = np.reshape(region_signal, (1, crop_size, crop_size))
        region_signal = np.repeat(region_signal, nb_pos, axis=0)
        region_bg = (1 - np.pad(
            d2, pad_bg, 'constant', constant_values=0)) * d3
        region_bg = region_bg.astype(bool)
        region_bg = np.reshape(region_bg, (1, crop_size, crop_size))
        region_bg = np.repeat(region_bg, nb_pos, axis=0)
        for i in range(nb_pos):
            crop = crops[i]
            bg = crop[region_bg[i]].mean()
            noise = crop[region_bg[i]].std()
            signal = crop[region_signal[i]].mean() - bg
            total_intensity = signal * region_signal[i].sum()
            val_bg.append(bg)
            val_noise.append(noise)
            val_signal.append(signal)
            val_total_intensity.append(total_intensity)
            nb_signal_pixels.append(region_signal[i].sum())
            nb_bg_pixels.append(region_bg[i].sum())
    elif mode == 'simple':
        flat_crops = np.sort(crops.reshape(-1, crop_size ** 2), axis=1)
        boundary_bg = flat_crops[:, int(crop_size ** 2 * bg_ratio)]
        boundary_bg = boundary_bg.reshape(-1, 1, 1)
        boundary_signal = flat_crops[:, ::-1][:,int(crop_size**2*signal_ratio)]
        boundary_signal = boundary_signal.reshape(-1, 1, 1)
        region_signal = crops > boundary_signal
        region_bg = crops < boundary_bg
        for i in range(nb_pos):
            crop = crops[i]
            crop_bg = crop[region_bg[i]]
            crop_signal = crop[region_signal[i]]
            bg, noise = np.mean(crop_bg), np.std(crop_bg)
            signal = np.mean(crop_signal) - bg
            val_bg.append(bg)
            val_signal.append(signal)
            val_noise.append(noise)
            val_total_intensity.append(signal * crop_signal.size)
            nb_signal_pixels.append(crop_signal.size)
            nb_bg_pixels.append(crop_bg.size)
    elif mode == 'adaptive':
        flat_crops = np.sort(crops.reshape(-1, crop_size ** 2), axis=1)
        boundary_bg = flat_crops[:, int(crop_size ** 2 * bg_ratio)]
        boundary_bg = boundary_bg.reshape(-1, 1, 1)
        region_bg = crops < boundary_bg
        region_signal = []
        for i in range(nb_pos):
            crop = crops[i]
            crop_bg = crop[region_bg[i]]
            bg, noise = np.mean(crop_bg), np.std(crop_bg)
            min_signal_val = bg + signal_thres * noise
            crop_signal = crop[crop > min_signal_val]
            if crop_signal.size > 0:
                signal = np.mean(crop_signal) - bg
            else:
                signal = 0
            region_signal.append(crop > min_signal_val)
            val_bg.append(bg)
            val_signal.append(signal)
            val_noise.append(noise)
            val_total_intensity.append(signal * crop_signal.size)
            nb_signal_pixels.append(crop_signal.size)
            nb_bg_pixels.append(crop_bg.size)
    else:
        raise ValueError('only support simple, rings, and adaptive mode.')

    val_bg = np.array(val_bg)
    val_signal = np.array(val_signal)
    val_noise = np.array(val_noise)
    val_total_intensity = np.array(val_total_intensity)
    nb_signal_pixels = np.array(nb_signal_pixels)
    nb_bg_pixels = np.array(nb_bg_pixels)
    snr = val_signal / val_noise

    if label_pixels:
        signal_pixels = None
        bg_pixels = None
        for i in range(nb_pos):
            signal_x, signal_y = np.where(region_signal[i] == 1)
            signal_x += (pos[i, 0] - half_crop)
            signal_y += (pos[i, 1] - half_crop)
            signal_pixels_ = np.concatenate(
                [
                    signal_x.reshape(-1, 1),
                    signal_y.reshape(-1, 1)
                ],
                axis=1
            )
            if signal_pixels is None:
                signal_pixels = signal_pixels_
            else:
                signal_pixels = np.concatenate(
                    [
                        signal_pixels,
                        signal_pixels_
                    ],
                    axis=0
                )

            bg_x, bg_y = np.where(region_bg[i] == 1)
            bg_x += (pos[i, 0] - half_crop)
            bg_y += (pos[i, 1] - half_crop)
            bg_pixels_ = np.concatenate(
                [
                    bg_x.reshape(-1, 1),
                    bg_y.reshape(-1, 1)
                ],
                axis=1
            )
            if bg_pixels is None:
                bg_pixels = bg_pixels_
            else:
                bg_pixels = np.concatenate(
                    [
                        bg_pixels,
                        bg_pixels_
                    ],
                    axis=0
                )
    else:
        signal_pixels = None
        bg_pixels = None

    snr_info = {
        'snr': snr,
        'total intensity': val_total_intensity,
        'signal values': val_signal,
        'background values': val_bg,
        'noise values': val_noise,
        'signal pixel num': nb_signal_pixels,
        'background pixel num': nb_bg_pixels,
        'signal pixels': signal_pixels,
        'background pixels': bg_pixels,
    }

    return snr_info


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
        frame = read_image(
            filepath,
            record['frame'],
            h5_obj=h5_obj,
            dataset=record['dataset']
        )
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
    elif hasattr(obj, '__iter__') and not isinstance(
            obj, (str, bytes, bytearray)
    ):
        size += sum([get_size(i, seen) for i in obj])
    return size


def collect_jobs(files, dataset, batch_size):
    jobs = []
    batch = []
    frames = 0
    print('collecting jobs...')
    for i in tqdm(range(len(files))):
        try:
            data_shape = get_data_shape(files[i])
            shape = data_shape[dataset]
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


def save_full_cxi(batch, cxi_file,
                  cxi_dtype='auto',
                  compression=None,
                  shuffle=True
                  ):
    """
    Save crystfel-compatible cxi file.
    :param batch: a list contains frame and peak info.
    :param cxi_file: output cxi filepath.
    :param cxi_dtype: datatype of cxi file.
    :param compression: compression filter used for raw data.
    :param shuffle: whether shuffle used for compression.
    :return: None.
    """
    print('saving %s' % cxi_file)
    nb_frame = len(batch)
    filepath_curr = None
    frames = []
    nb_peaks = np.zeros(nb_frame, dtype=np.int)
    peaks_x = np.zeros((nb_frame, 1024))
    peaks_y = np.zeros((nb_frame, 1024))
    peaks_intensity = np.zeros((nb_frame, 1024))
    peaks_snr = np.zeros((nb_frame, 1024))
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
        frame = read_image(filepath, record['frame'],
                           h5_obj=h5_obj,
                           dataset=record['dataset'])
        peak_info = record['peak_info']
        nb_peak = min(len(peak_info['snr']), 1024)
        nb_peaks[i] = nb_peak
        peaks_x[i, :nb_peak] = peak_info['pos'][:nb_peak, 0]
        peaks_y[i, :nb_peak] = peak_info['pos'][:nb_peak, 1]
        peaks_intensity[i, :nb_peak] = peak_info['total intensity'][:nb_peak]
        peaks_snr[i, :nb_peak] = peak_info['snr'][:nb_peak]
        frames.append(frame)

    in_dtype = frame.dtype
    if cxi_dtype == 'auto':
        cxi_dtype = in_dtype
    else:
        cxi_dtype = np.dtype(cxi_dtype)
    frames = np.array(frames).astype(cxi_dtype)
    n, x, y = frames.shape
    if os.path.exists(cxi_file):
        os.rename(cxi_file, '%s.bk' % cxi_file)
    f = h5py.File(cxi_file, 'w')
    # save patterns
    f.create_dataset(
        'data',
        shape=(n, x, y),
        dtype=cxi_dtype,
        data=frames,
        compression=compression,
        chunks=(1, x, y),
        shuffle=shuffle,
    )
    # save peak info
    f.create_dataset('peak_info/nPeaks', data=nb_peaks)
    f.create_dataset('peak_info/peakXPosRaw', data=peaks_y)  # fs
    f.create_dataset('peak_info/peakYPosRaw', data=peaks_x)  # ss
    f.create_dataset('peak_info/peakTotalIntensity', data=peaks_intensity)
    f.create_dataset('peak_info/peakSNR', data=peaks_snr)
    f.close()


def fit_ellipse(x, y):
    """
    Fit ellipse to scattered data with Direct Least Squares Fitting Method.
    DOI: 10.1109/ICPR.1996.546029
    :param x: 1d array
    :param y: 1d array
    :return: ellipse parameters, including a, b, h, k, tao
    """
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError('x, y must have the same length.')
    nb_points = x.size
    D = np.zeros((nb_points, 6))
    D[:, 0] = x ** 2
    D[:, 1] = x * y
    D[:, 2] = y ** 2
    D[:, 3] = x
    D[:, 4] = y
    D[:, 5] = 1
    S = D.T.dot(D)
    C = np.zeros((6, 6), dtype=np.float)
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    eig_val, eig_vec = eig(inv(S).dot(C))
    idx = np.where(eig_val.real > 0)[0][0]
    solution = eig_vec[:, idx]
    a, b, c, d, e, f = solution  # ax**2 + bxy + cy**2 + dx + ey + f = 0
    # calculate ellipse parameters
    M0 = np.array([[f, d/2., e/2.],
                   [d/2., a, b/2.],
                   [e/2., b/2., c]])
    M = np.array([[a, b/2.],
                  [b/2., c]])
    eig_val, _ = eig(M)
    l1, l2 = eig_val.real
    if abs(l1 - a) > abs(l1 - c):
        l1, l2 = l2, l1

    ellipse = {
        'a': math.sqrt(-det(M0)/det(M)/l1),
        'b': math.sqrt(-det(M0)/det(M)/l2),
        'h': (b*e - 2*c*d)/(4*a*c - b**2),
        'k': (b*d - 2*a*e)/(4*a*c - b**2),
        'tao': math.atan(b/(a-c))/2.
    }
    return ellipse


def fit_circle(x, y, tol=3.0, init_center=(0, 0), init_radius=1.):
    """
    Fit circle to scattered points with specified tolerance.
    :param x: x coordinates, 1d array.
    :param y: y coordinates, 1d array.
    :param tol: fitting tolerance in sigma.
    :param init_center: initial center for optimization, 2 elements array.
    :param init_radius: initial center for optimization.
    :return: circle parameters, including center, radius, radius_std,
             radius_min, radius_max, fitting points num
    """
    x, y = np.array(x), np.array(y)
    center = np.array(init_center).reshape(-1)
    radius = init_radius
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')
    if center.size != 2:
        raise ValueError('init_center must have 2 elements: x, y.')
    while True:
        def target(variables):
            x0, y0, r = variables
            return np.mean(
                (np.sqrt(
                    (x - x0) ** 2 + (y - y0) ** 2
                ) - r) ** 2
            )

        in_vars = np.array([center[0], center[1], radius])
        ret = minimize(target, in_vars, method='CG')
        center = ret.x[0:2]
        radii = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        radii_mean = radii.mean()
        radii_std = radii.std()
        valid_idx = np.where(np.abs(radii - radii_mean) < tol * radii_std)[0]
        if len(valid_idx) == x.size:
            break
        elif len(x) <= 10:  # exit if not enough points to fit
            break
        else:
            x = x[valid_idx]
            y = y[valid_idx]
    circle = {
        'center': center,
        'radius': radii_mean,
        'radius_std': radii_std,
        'radius_min': radii.min(),
        'radius_max': radii.max(),
        'fitting peaks num': len(x)
    }
    return circle


def get_photon_wavelength(photon_energy):
    """
    Calucate wavelength for given photon energy in eV.
    :param photon_energy: photon energy in eV.
    :return: photon wavelength in angstrom
    """
    h = 4.135667E-15  # in eV/s
    c = 2.99792458E8  # in m/s
    wavelength = h*c/photon_energy
    wavelength *= 1.E10
    return wavelength


def get_photon_energy(photon_wavelength):
    """
    Calculate photon energy for give wavelength in angstrom.
    :param photon_wavelength: photon wavelength in angstrom.
    :return: photon energy in eV.
    """
    h = 4.135667E-15  # in eV/s
    c = 2.99792458E8  # in m/s
    photon_wavelength *= 1.E-10
    photon_energy = h*c/photon_wavelength
    return photon_energy


def build_grid_image(dim0, dim1):
    idx, idy = np.indices((dim0, dim1))
    image = np.zeros((dim0, dim1))
    image[(idx+idy) % 2 == 0] = 1
    return image
