import h5py
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import square, disk


def read_image(filepath, frame=0, h5_obj=None, h5_dataset=None):
    ext = filepath.split('.')[-1]
    if ext == 'npy':
        data = np.load(filepath)
    elif ext in ('h5', 'cxi'):
        if len(h5_obj[h5_dataset].shape) == 3:
            data = h5_obj[h5_dataset][frame].value
        else:
            data = h5_obj[h5_dataset].value
    return data


def find_peaks(image,
               mask=None,
               gaussian_sigma=1.,
               min_gradient=0.,
               min_distance=0,
               refine=False,
               max_peaks=500,
               min_snr=0.,
               ):
    raw_image = image.copy()
    if gaussian_sigma >= 0:
        image = gaussian_filter(image.astype(np.float32), gaussian_sigma)
    grad = np.gradient(image.astype(np.float32))
    grad_mag = np.sqrt(grad[0]**2. + grad[1]**2.)
    peaks = peak_local_max(grad_mag,
                           min_distance=int(round((min_distance-1.)/2.)),
                           threshold_abs=min_gradient, num_peaks=max_peaks)
    # mask out unvalid peaks
    if mask is not None:
        valid_peak_ids = []
        for i in range(peaks.shape[0]):
            peak = np.round(peaks[i].astype(np.int))
            if mask[peak[0], peak[1]] == 1:
                valid_peak_ids.append(i)
        peaks = peaks[valid_peak_ids]

    # refine peak location
    if refine:
        opt_peaks = peaks.copy().astype(np.float32)
        for i in range(peaks.shape[0]):
            x, y = peaks[i]
            crop = raw_image[x-4:x+5, y-4:y+5].astype(np.float32)
            crop_1d = np.sort(crop.flatten())
            crop_1d_smooth = np.convolve(
                crop_1d, np.ones(3), mode='same')
            grad = np.gradient(crop_1d_smooth)
            thres = crop_1d[np.argmax(grad)]
            signal_mask = (crop >= thres).astype(np.int)
            ids = (np.indices((9, 9)) - 4).astype(np.float)
            weight = np.sum(crop*signal_mask)
            opt_peaks[i, 0] += np.sum(crop*ids[0]*signal_mask) / weight
            opt_peaks[i, 1] += np.sum(crop*ids[1]*signal_mask) / weight
        peaks = opt_peaks

    # remove weak peak
    strong_peak_ids = []
    for i in range(len(peaks)):
        x, y = np.round(opt_peaks[i]).astype(np.int)
        if x-3 < 0 or x+4 > raw_image.shape[0]:
            continue
        elif y-3 < 0 or y+4 > raw_image.shape[1]:
            continue
        ROI = raw_image[x-3:x+4, y-3:y+4]
        snr = calc_snr(ROI)
        if snr > min_snr:
            strong_peak_ids.append(i)
    peaks = peaks[strong_peak_ids]

    return peaks


def calc_snr(ROI, signal_radius=1, noise_inner_radius=2, noise_outer_radius=3):
    assert noise_outer_radius > noise_inner_radius > signal_radius
    assert ROI.shape[0] == ROI.shape[1] == noise_outer_radius*2+1
    d1 = disk(signal_radius)
    d2 = disk(noise_inner_radius)
    d3 = disk(noise_outer_radius)
    pad_signal = noise_outer_radius - signal_radius
    pad_noise = noise_outer_radius - noise_inner_radius
    region_signal = np.pad(d1, pad_signal, 'constant', constant_values=0)
    region_noise = (1-np.pad(d2, pad_noise, 'constant', constant_values=0))*d3
    val_signal = np.sum(ROI*region_signal) / np.sum(region_signal)
    val_noise = np.sum(ROI*region_noise) / np.sum(region_noise)
    snr = val_signal / val_noise
    return snr


def get_h5_info(filepath):
    f = h5py.File(filepath, 'r')
    keys = []

    def _get_all_dataset(key):
        if isinstance(f[key], h5py._hl.dataset.Dataset):
            keys.append(key)
    f.visit(_get_all_dataset)
    data_info = []
    for i in range(len(keys)):
        key = keys[i]
        if len(f[key].shape) in (2, 3):
            data_info.append({'key': key, 'shape': f[key].shape})
    f.close()
    return data_info
