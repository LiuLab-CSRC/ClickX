import h5py
from skimage.feature import peak_local_max


def read_image(filepath):
    dataset_name = 'data/rawdata0'
    f = h5py.File(filepath, 'r')
    data = f[dataset_name].value
    return data


def eval_image(image, min_intensity, min_distance):
    peaks = peak_local_max(image, min_distance=min_distance, 
                           threshold_abs=min_intensity)
    return peaks
