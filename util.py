import h5py
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve1d
from skimage.morphology import disk


def read_image(filepath, frame=0, h5_obj=None, h5_dataset=None):
    ext = filepath.split('.')[-1]
    if ext == 'npy':
        data = np.load(filepath)
    elif ext in ('h5', 'cxi'):
        if len(h5_obj[h5_dataset].shape) == 3:
            data = h5_obj[h5_dataset][frame].value
        else:
            data = h5_obj[h5_dataset].value
    else:
        print('Unsupport fomat: %s' % ext)
        return None
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
    grad_mag = np.sqrt(grad[0] ** 2. + grad[1] ** 2.)
    peaks = peak_local_max(grad_mag,
                           min_distance=int(round((min_distance - 1.) / 2.)),
                           threshold_abs=min_gradient, num_peaks=max_peaks)
    # mask out invalid peaks
    if mask is not None:
        valid_peak_ids = []
        for i in range(peaks.shape[0]):
            peak = np.round(peaks[i].astype(np.int))
            if mask[peak[0], peak[1]] == 1:
                valid_peak_ids.append(i)
        peaks = peaks[valid_peak_ids]
    # refine peak location
    if refine:
        peaks = refine_peaks(raw_image, peaks)
    # remove weak peak
    crops = []
    for i in range(len(peaks)):
        x, y = np.round(peaks[i]).astype(np.int)
        if x - 3 < 0 or x + 4 > raw_image.shape[0]:
            continue
        elif y - 3 < 0 or y + 4 > raw_image.shape[1]:
            continue
        crops.append(raw_image[x - 3:x + 4, y - 3:y + 4])
    crops = np.array(crops)
    snr = calc_snr(crops)
    peaks = peaks[snr >= min_snr]
    return peaks


def refine_peaks(image, peaks):
    opt_peaks = peaks.copy().astype(np.float32)
    peaks = np.round(peaks).astype(np.int)
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
    grad = np.gradient(crop_1ds_smooth)[1]
    max_ids = np.argmax(grad, axis=1)
    signal_masks = []
    for i in range(len(max_ids)):
        signal_masks.append((crops[i] >= crop_1ds[i, max_ids[i]]).astype(np.int))
    signal_masks = np.array(signal_masks)
    ids = (np.indices((9, 9)) - 4).astype(np.float)
    weight = np.sum(np.sum(crops * signal_masks, axis=1), axis=1).astype(np.float32)
    opt_peaks[:, 0] += np.sum(np.sum(crops * signal_masks * ids[0], axis=1), axis=1) / weight
    opt_peaks[:, 1] += np.sum(np.sum(crops * signal_masks * ids[1], axis=1), axis=1) / weight
    return opt_peaks


def calc_snr(crops, signal_radius=1, noise_inner_radius=2, noise_outer_radius=3):
    d1 = disk(signal_radius)
    d2 = disk(noise_inner_radius)
    d3 = disk(noise_outer_radius)
    pad_signal = noise_outer_radius - signal_radius
    pad_noise = noise_outer_radius - noise_inner_radius
    region_signal = np.pad(d1, pad_signal, 'constant', constant_values=0)
    region_noise = (1 - np.pad(d2, pad_noise, 'constant', constant_values=0)) * d3
    val_signal = np.sum(np.sum(crops * region_signal, axis=1), axis=1) / np.sum(region_signal)
    val_noise = np.sum(np.sum(crops * region_noise, axis=1), axis=1) / np.sum(region_noise)
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


def main():
    f = h5py.File('/Volumes/LaCie/data/temp/data1/LCLS_2014_Feb02_r0137_091107_6576.h5', 'r')
    img = f['data/rawdata0'].value
    find_peaks(img, gaussian_sigma=1., min_gradient=200, min_distance=10, min_snr=4., refine=True, max_peaks=1000)


if __name__ == '__main__':
    main()
