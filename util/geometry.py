# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import h5py
import os


def det2fourier(pos, wavelength, det_dist):
    """
    Convert detector coordinates to fourier coordinates.
    :param pos: detector coordinates, Nx2 array.
    :param wavelength: wavelength in angstrom.
    :param det_dist: detector distance in meters.
    :return: q vectors in fourier space, Nx3 array.
    """
    pos = np.array(pos)
    if len(pos.shape) != 2:
        raise ValueError('pos should be 2d array!')
    if pos.shape[1] != 2:
        raise ValueError('pos should have Nx2 shape!')
    det_dist = np.reshape(np.ones(pos.shape[0]) * det_dist, (-1, 1))
    q1 = np.hstack((pos, det_dist))
    q1_norm = norm(q1, axis=1).reshape(-1, 1)
    q1 /= q1_norm
    q0 = np.asarray([0., 0., 1.])
    q0 = q0.reshape((1, -1)).repeat(pos, axis=0)
    q = 1. / wavelength * (q1 - q0)
    return q


def get_hkl(q, trans_matrix=None, trans_matrix_inv=None):
    """
    Convert q vectors in fourier space to miller indices.
    :param q: fourier coordinates, Nx3 array.
    :param trans_matrix: transform matrix, 3x3 array.
    :param trans_matrix_inv: inverse transform matrix, 3x3 array.
    :return: miller indices, Nx3 array.
    """
    q = np.array(q)
    if trans_matrix is None and trans_matrix_inv is None:
        raise ValueError(
            'Either trans_matrix or trans_matrix_inv should be specified!')
    if len(q.shape) != 2:
        raise ValueError('q should be 2d array!')
    if q.shape[1] != 2:
        raise ValueError('q should have Nx2 shape!')
    if trans_matrix is not None:
        trans_matrix_inv = np.linalg.inv(np.array(trans_matrix))
    elif trans_matrix_inv is not None:
        trans_matrix_inv = np.array(trans_matrix_inv)
    hkl = trans_matrix_inv.dot(q.T).T
    return hkl


def load_geom(filepath):
    """
    Load geometry file.
    :param filepath: path of geometry file.
    :return: x, y, z coordinates of geometry data.
    """
    ext = os.path.splitext(filepath)[1]
    if ext == '.h5':
        f = h5py.File(filepath, 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.geom':
        from psgeom import camera
        cspad = camera.Cspad.from_crystfel_file(filepath)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.psana':
        from psgeom import camera
        cspad = camera.Cspad.from_psana_file(filepath)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    else:
        raise ValueError('geom file should be h5(cheetah), psana(psana), '
                         'or geom(crystfel) format')


class Geometry(object):
    def __init__(self, geom_file, pixel_size):
        """
        Geometry class.
        :param geom_file: filepath of geometry file.
        :param pixel_size: pixel size in micrometers.
        """
        self.geom_file = geom_file
        self.pixel_size = pixel_size * 1.E-6
        self.geom_x, self.geom_y, self.geom_z = load_geom(geom_file)
        self.x = (np.round(self.geom_x / self.pixel_size)).astype(int)
        self.y = (np.round(self.geom_y / self.pixel_size)).astype(int)
        self.offset = np.array([self.x.min(), self.y.min()])
        self.x -= self.offset[0]
        self.y -= self.offset[1]
        self.shape = self.x.max() + 1, self.y.max() + 1

    def rearrange(self, raw_img):
        """
        Rearrange raw image to assembled image.
        :param raw_img: raw image, 2d array.
        :return: assembled image, 2d array.
        """
        assembled_img = np.zeros(self.shape)
        assembled_img[self.x.reshape(-1),
                      self.y.reshape(-1)] = raw_img.reshape(-1)
        return assembled_img

    def map(self, raw_pos):
        """
        Map raw position to assembled position in pixels.
        :param raw_pos: raw position in pixels, Nx2 shape.
        :return: assembled position in pixels, Nx2 shape.
        """
        raw_pos = (np.round(raw_pos)).astype(int)
        if len(raw_pos.shape) != 2:
            raise ValueError('raw_pos should be 2d array!')
        if raw_pos.shape[1] != 2:
            raise  ValueError('raw_pos should have Nx2 shape!')
        # map raw coordinates to assembled coordinates in meters
        assembled_pos = np.zeros_like(raw_pos, dtype=float)
        assembled_pos[:, 0] = self.geom_x[raw_pos[:, 0], raw_pos[:, 1]]
        assembled_pos[:, 1] = self.geom_y[raw_pos[:, 0], raw_pos[:, 1]]
        # map assembled coordinates in meters to pixels
        assembled_pos /= self.pixel_size
        assembled_pos -= self.offset
        return assembled_pos
