import h5py
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter, convolve1d
from skimage.morphology import disk


def read_image(filepath, frame=0, h5_obj=None, dataset=None):
    ext = filepath.split('.')[-1]
    if ext == 'npy':
        data = np.load(filepath)
    elif ext  == 'npz':
        data = np.load(filepath)[dataset]
    elif ext in ('h5', 'cxi'):
        if len(h5_obj[dataset].shape) == 3:
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
