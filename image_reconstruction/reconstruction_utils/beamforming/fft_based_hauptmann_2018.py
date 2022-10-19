"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Jenni Poimala
SPDX-FileCopyrightText: 2022 Andreas Hauptmann
SPDX-License-Identifier: MIT

"""

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import math

import copy


def fft_hauptmann_2d(time_series_data, detection_elements, sampling_rate_hz,
                     field_of_view, spacing_m, speed_of_sound_in_m_per_s):
    """
    Implementation of an FFT-based reconstruction algorithm.
    
    The implementation reflects the reconstruction algorithm described by Hauptmann et al., 2018.
    Additionally, original papers are also cited:

        Hauptmann, A., Cox, B., Lucka, F., Huynh, N., Betcke, M., Beard, P., & Arridge, S. 
        (2018, September). Approximate k-space models and deep learning for fast photoacoustic 
        reconstruction. In International Workshop on Machine Learning for Medical Image Reconstruction 
        (pp. 103-111). Springer, Cham.
        
        Köstli, K. P., Frenz, M., Bebie, H., & Weber, H. P. (2001). Temporal backward projection 
        of optoacoustic pressure transients using Fourier transform methods. Physics in Medicine 
        & Biology, 46(7), 1863.

        Xu, Y., Feng, D., & Wang, L. V. (2002). Exact frequency-domain reconstruction for 
        thermoacoustic tomography. I. Planar geometry. IEEE transactions on medical imaging, 
        21(7), 823-828.

    Parameters
    ----------
    time_series_data: np.ndarray
        A 2D numpy array with the following internal array definition: [detectors, time samples]
    detection_elements: dict
        Definition of the transducer elements
    sampling_rate_hz: float
        Data sampling rate in Hz
    field_of_view: np.ndarray
        The target field of view in [xmin, xmax, ymin, ymax, zmin, zmax] in meters
    spacing_m: float
        The target resolution in units of meters; default=0.0001
    speed_of_sound_in_m_per_s: float
        The speed of sound in units of meters

    Returns
    -------
    np.ndarray
        Returns a numpy array with the reconstruction result in the format: [xdim, 1, zdim]

    """

    time_spacing_in_s = 1.0 / sampling_rate_hz
    
    positions = detection_elements["positions"]

    if field_of_view is None:

        # find the axis of the planar sensor 
        sensor_axis = [i for i, e in enumerate(
            [len(set(positions[:, 0])), len(set(positions[:, 1])), len(set(positions[:, 2]))]) if e != 1]

        if len(sensor_axis) > 1:
            print('ERROR: Sensor is not a planar sensor')

        # make grid for the sensor axis 
        x = np.arange(positions[:, sensor_axis[0]].min(), positions[:, sensor_axis[0]].max()+spacing_m, spacing_m)

        # sensor locations
        x_sensor = positions[:, sensor_axis[0]]

        # locate sensors to the sensor axis grid that is find nearest pixel   
        idxs = find_nearest_index(x, x_sensor)

        # create a new zero pressure data set       
        p_t = np.zeros((x.shape[0], time_series_data.shape[1]))

        # put the measured data into correct location      
        p_t[idxs] = time_series_data

        # Use all data points      
        nt_factor = 1

        # spacing      
        dx = spacing_m
        dy = dx
        
        # speed of sound
        c = speed_of_sound_in_m_per_s
        
        # time spacing
        dt = time_spacing_in_s

        # #########################################################################
        # FFT based reconstruction
        # #########################################################################

        # mirror the time domain data
        p_t = np.transpose(p_t)
         
        p_up = np.flipud(p_t)
        p0 = np.concatenate((p_up, p_t[1::, :]), 0)
        
        size = p0.shape
        
        nt = size[0]
        ny = size[1]

        output = np.zeros((ny, 1, math.floor(nt/nt_factor/2)))

        # compute kgrids
        kgrid_back = KGrid2D(math.ceil(nt / nt_factor), dx, ny, dy, c)

        kgrid_back_c = KGrid2D(nt, dx, ny, dy, dx / dt)

        # computational grid for w and w_new
        c = kgrid_back_c.c
        w = c*kgrid_back_c.kx
        w_new = kgrid_back.c*kgrid_back.k

        # scalig factor
        sf = np.square(w/c) - np.square(kgrid_back_c.ky)
        sf = c*c*np.sqrt(sf.astype(np.complex))
        sf = np.divide(sf, 2*w)

        idx1 = np.where((w == 0) & (kgrid_back_c.ky == 0))
        sf[idx1] = c/2

        idx2 = np.where(np.abs(w) < np.abs(c*kgrid_back_c.ky) )
        sf[idx2] = 0

        idx = np.where(np.isnan(sf))
        sf[idx] = 0

        # computational grids needed for interpolation
        ky = kgrid_back_c.ky_vec
        ky_i = kgrid_back.ky

        w = c*kgrid_back_c.kx_vec
        w_i = w_new

        points_inv = (w.flatten(), ky.flatten())
        points_i_inv = (w_i.flatten(), ky_i.flatten())

        nxyz_i = w_i.shape

        # indexes for correct part of reconstruction
        ind_s = math.ceil((p0.shape[0]+1)/2/nt_factor)
        ind_e = math.floor((p0.shape[0])/nt_factor)+1

        # the FFT of the input data and scaling
        p0 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(p0)))
        p0 = p0*sf

        # interpolation   
        gi = RegularGridInterpolator(points_inv, p0, bounds_error=False, fill_value=0)
        p0 = gi(points_i_inv)
        p0 = np.reshape(p0, (nxyz_i[0], nxyz_i[1]))

        # the inverse FFT
        p0 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(p0)))

        # take correct part of reconstruction and scale
        p0 = 2*2*p0.real[ind_s:ind_e, :]/c / nt_factor

        # shape result to desired output
        p0 = np.transpose(p0)
        output[:, :, :] = p0[:, None, :]

    else:

        field_of_view_voxels = np.round(field_of_view / spacing_m).astype(int)

        x_dim = (field_of_view_voxels[1] - field_of_view_voxels[0])
        y_dim = (field_of_view_voxels[3] - field_of_view_voxels[2])
        z_dim = (field_of_view_voxels[5] - field_of_view_voxels[4])

        # Just to make sure not to allocate a 0-dimensional array
        if x_dim < 1:
            x_dim = 1
        if y_dim < 1:
            y_dim = 1
        if z_dim < 1:
            z_dim = 1

        dim = [x_dim, y_dim, z_dim]
        
        # find the axis of the planar sensor 
        sensor_axis = [i for i, e in enumerate(
            [len(set(positions[:, 0])), len(set(positions[:, 1])), len(set(positions[:, 2]))]) if e != 1]

        if len(sensor_axis) > 1:
            print('ERROR: Sensor is not a planar sensor')

        # make grid for the sensor axis 
        x = np.arange(field_of_view[2*sensor_axis[0]],  field_of_view[2*sensor_axis[0]+1], spacing_m)

        # sensor locations
        x_sensor = positions[:, sensor_axis[0]]

        # locate sensors to the sensor axis grid that is find nearest pixel   
        idxs = find_nearest_index(x, x_sensor)

        # create a new zero pressure data set       
        p_t = np.zeros((x.shape[0], time_series_data.shape[1]))

        # put the measured data into correct location      
        p_t[idxs] = time_series_data

        # determine NtFactor = relation between number of pixels in depht direction and number of time points
        dim_am = copy.deepcopy(dim)
        dim_am.remove(dim_am[sensor_axis[0]])

        # Use only part of data points
        nt_factor = math.floor(time_series_data.shape[1]/max(dim_am))

        # spacing
        dx = spacing_m
        dy = dx
        
        # speed of sound
        c = speed_of_sound_in_m_per_s
        
        # time spacing
        dt = time_spacing_in_s

        # #########################################################################
        # FFT based reconstruction
        # #########################################################################

        # mirror the time domain data 
        p_t = np.transpose(p_t)

        p_up = np.flipud(p_t)
        p0 = np.concatenate((p_up, p_t[1::, :]), 0)

        size = p0.shape

        nt = size[0]
        ny = size[1]

        output = np.zeros((dim))

        # compute kgrids
        kgrid_back = KGrid2D(math.ceil(nt / nt_factor), dx, ny, dy, c)
        kgrid_back_c = KGrid2D(nt, dx, ny, dy, dx / dt)

        # computational grid for w and w_new
        c = kgrid_back_c.c
        w = c*kgrid_back_c.kx
        w_new = kgrid_back.c*kgrid_back.k

        # scalig factor
        sf = np.square(w/c) - np.square(kgrid_back_c.ky)
        sf = c*c*np.sqrt(sf.astype(np.complex))

        sf = np.divide(sf, 2*w)

        idx1 = np.where((w == 0) & (kgrid_back_c.ky == 0))
        sf[idx1] = c/2
         
        idx2 = np.where(np.abs(w) < np.abs(c*kgrid_back_c.ky))
        sf[idx2] = 0
              
        idx = np.where(np.isnan(sf))
        sf[idx] = 0

        # computational grids needed for interpolation
        ky = kgrid_back_c.ky_vec
        ky_i = kgrid_back.ky

        w = c*kgrid_back_c.kx_vec
        w_i = w_new

        points_inv = (w.flatten(), ky.flatten())
        points_i_inv = (w_i.flatten(), ky_i.flatten())

        nxyz_i = w_i.shape

        # indexes for correct part of reconstruction
        ind_s = math.ceil((p0.shape[0]+1)/2/nt_factor)
        ind_e = math.floor((p0.shape[0])/nt_factor)+1

        # the FFT of the input data and scaling
        p0 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(p0)))
        p0 = p0*sf
   
        # interpolation
        gi = RegularGridInterpolator(points_inv, p0, bounds_error=False, fill_value=0)
        p0 = gi(points_i_inv)
        p0 = np.reshape(p0, (nxyz_i[0], nxyz_i[1]))

        # the inverse FFT
        p0 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(p0)))

        # take correct part of reconstruction and scale
        p0 = 2*2*p0.real[ind_s:ind_e, :]/c / nt_factor

        p0 = np.transpose(p0)

        # take only part that corrensponds to field of view
        p0 = p0[0:dim[0], 0:dim[2]]

        # shape result to desired shape
        output[:, :, :] = np.reshape(p0, (dim))

    return output


class KGrid2D(object):
    def __init__(self, n_x, dx, n_y, dy, c):

        [k, kx, ky, kx_vec, ky_vec] = make_kgrid_2d(n_x, dx, n_y, dy)

        self._kx = kx
        self._ky = ky
        self._k = k
        
        self._kx_vec = kx_vec
        self._ky_vec = ky_vec

        self.Nx = n_x
        self.Ny = n_y
        self.dx = dx
        self.dy = dy
        self.c = c

    @property
    def kx(self):        
        return self._kx

    @property
    def ky(self):        
        return self._ky

    @property
    def k(self):        
        return self._k

    @property
    def kx_vec(self):        
        return self._kx_vec

    @property
    def ky_vec(self):        
        return self._ky_vec


def make_kgrid_2d(n_x, dx, n_y, dy):
    
    kx_vec = make_dim(n_x, dx)
    ky_vec = make_dim(n_y, dy)

    kx = np.tile(kx_vec.reshape(-1, 1), (1, n_y))
    ky = np.tile(ky_vec, (n_x, 1))

    k = np.zeros([n_x, n_y])
    k = (kx_vec**2).reshape(-1, 1)+k
    k = (ky_vec**2) + k
    k = np.sqrt(k)
                   
    return k, kx, ky, kx_vec, ky_vec 


def make_dim(n_x, dx):
    
    if (n_x % 2) == 0:
        nx = np.arange(-n_x / 2, n_x / 2) / n_x
    else:
        nx = np.arange(-(n_x - 1) / 2, n_x / 2) / n_x
        
    # force middle value to be zero in case 1/n_x is a recurring number and the series doesn't give exactly zero
    nx[math.floor(n_x / 2)] = 0
            
    # define the wavenumber vector components
    kx_vec = (2*np.pi/dx)*nx
    
    return kx_vec


def find_nearest_index(array, value):
    
    idxs = []
    
    for ii in value:
        
        array = np.asarray(array)
    
        idx = (np.abs(array - ii)).argmin()
        
        idxs.append(idx)

    return idxs
