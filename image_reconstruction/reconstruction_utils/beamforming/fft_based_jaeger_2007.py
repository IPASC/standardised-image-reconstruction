"""
SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Mengjie Shi
SPDX-FileCopyrightText: 2022 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from math import ceil, floor, pi
from scipy.ndimage import zoom
import numpy as np


def fft_based_jaeger_2d(time_series_data,
                        detection_elements,
                        sampling_rate,
                        sos: float,
                        delay,
                        zero_pad_detectors,
                        zero_pad_time,
                        fourier_coefficients_dim,
                        spacing_m,
                        field_of_view):
    """
    Implementation of an FFT-based algorithm.
    
    The implementation reflects the reconstruction algorithm described by Jaeger et al., 2007:

      Jaeger, M., Schüpbach, S., Gertsch, A., Kitz, M., & Frenz, M. (2007). Fourier reconstruction 
      in optoacoustic imaging using truncated regularized inverse k-space interpolation. Inverse Problems, 
      23(6), S51.
      
    Parameters
    ----------

    :param time_series_data: A 2D numpy array with the following internal array definition:
                            [detectors, time samples]
    :param detection_elements: transducer elements
    :param sampling_rate: data sampling rate (AcousticX 40MHz)
    :param sos: speed of sound in units of meters
    :param delay: time delay from laser irradiation to signal acquisition start (default 0)
    :param zero_pad_detectors: 1=zero pad in lateral (X) direction; 0=don't typically 1
    :param zero_pad_time: 1=zero pad in axial (t,time) direction; 0=don't typically 1
    :param fourier_coefficients_dim: signal fourier coefficients a single image fourier
        coefficient is interploated (5)
    :param spacing_m: The target resolution in meters; default=0.0001
    :param field_of_view: The target field of view in [xmin, xmax, ymin, ymax, zmin, zmax] in meters

    Returns
    ----------
    :return: reconstructed_image: resulting image, has equal number of samples in axial(z) direction as the signal.
             The number of image lines is: (number of elements -1) *samplingX +1

    """

    # Extract pitch from detection elements
    pitch_x_y_z = np.abs(detection_elements["positions"][1] - detection_elements["positions"][0])
    pitch = np.max(pitch_x_y_z)

    # Caution! This assumed that we have a linear uniform progression of
    # the elements along one axis only!

    # DATA DIMENSIONS
    # dimension of the signal and image in number of samples
    num_detectors = time_series_data.shape[0]  # check X is 128
    num_z_samples = time_series_data.shape[1]
    num_time_samples = num_z_samples

    # corresponding physical dimension of signal and image in [mm]
    image_extent_x_mm = num_detectors * pitch * 10 ** 3

    # These two should always be the same, no?
    image_extent_z_mm = num_z_samples * sos * 10 ** (-3) / (sampling_rate * 10 ** (-6))
    time_extent_mm = num_time_samples * sos * 10 ** (-3) / (sampling_rate * 10 ** (-6))

    # zero padding of the signal data in num_time_samples-direction, by a factor of two
    # this measure reduces aliasing artifacts when a strong OA source is located either
    # close to the start or to the end of the image frame
    # generally the influence is quite low, if fourier_coefficients_dim is set to a large value such as 5.
    if zero_pad_time:
        # the signal matrix is padded to double num_time_samples-size
        time_series_data = np.append(time_series_data, np.zeros((num_detectors, num_time_samples)), axis=1)
        # dimensions of image frame are doubled
        num_z_samples = 2 * num_z_samples
        image_extent_z_mm = 2 * image_extent_z_mm
        # dimensions of signal frame are doubled
        num_time_samples = 2 * num_time_samples
        time_extent_mm = 2 * time_extent_mm

    # zero padding of the signal data in X-direction, by a factor of two
    # this measure reduces aliasing artifacts when a strong OA source is located close
    # to either side of the image frame, if a source is located outside the image frame,
    # it may be correctly reconstructed, leading to less artifacts.
    if zero_pad_detectors:
        half_num_detectors = round(num_detectors / 2)  # check if round to the nearest integer (floating)
        delta_x_extent = image_extent_x_mm / 2
        # dimensions of image field are extended
        time_series_data = np.concatenate((np.zeros((half_num_detectors, num_time_samples)),
                                           time_series_data,
                                           np.zeros((half_num_detectors, num_time_samples))), axis=0)
        num_detectors = num_detectors + 2 * half_num_detectors
        image_extent_x_mm = image_extent_x_mm + 2 * delta_x_extent

    # RECONSTRUCTION ALGORITHM
    # 2D array containing kx and kz values of k-vectors of the image fourier space
    [k_x, k_z] = np.meshgrid(
        np.arange(-ceil((num_detectors - 1) / 2), floor((num_detectors - 1) / 2) + 1) / image_extent_x_mm,
        np.arange(-ceil((num_z_samples - 1) / 2), floor((num_z_samples - 1) / 2) + 1) / image_extent_z_mm)
    kz = k_z.T
    kx = k_x.T
    # 2D array containing kt values of k-vectors of the signal fourier space
    kt = kz

    # projection of (kx,kz) onto (kx,kt) describing the signal formation in the fourier space
    # projection of kz onto kt
    kt2 = -np.sqrt(kz ** 2 + kx ** 2)  # check math.sqrt， power of each elememt
    # calculate the jacobinate of the projection of kz onto kt

    kt2[(kz == 0) & (kx == 0)] = 1  # &
    jacobinate = kz / kt2
    jacobinate[(kz == 0) & (kx == 0)] = 1
    kt2[(kz == 0) & (kx == 0)] = 0

    # if kt2 out of bound of kt, take the modulo value
    sample_frequency = num_time_samples / time_extent_mm
    kt2 = (kt2 + sample_frequency / 2) % sample_frequency - sample_frequency / 2

    # 2D fourier spectrum of signal data
    sigtrans = np.fft.fft2(time_series_data)
    sigtrans = np.fft.fftshift(sigtrans)

    # only negatively propagating waves are detected,
    sigtrans[kz > 0] = 0

    # initialize array for image fourier coefficients
    ptrans = np.zeros((num_detectors, num_z_samples)).astype(complex)

    # numbers of signal fourier coefficients used for interpolation of image
    # fourier coefficient, in up and down direction
    num_fourier_coeff_up = ceil((fourier_coefficients_dim - 1) / 2)
    num_fourier_coeff_down = floor((fourier_coefficients_dim - 1) / 2)

    # helper index vector covering the range of fourier amplitudes going into interpolation
    ktrange = np.arange(-num_fourier_coeff_down, num_fourier_coeff_up + 1)  # vector (5,)

    # loop for interpolation of image fourier coefficients from signal fourier coefficients
    # for each image line do
    for xind in range(0, num_detectors):
        # calculate kt-index into kt-dimension of signal fourier space, from the kt value resulting
        # from the projection of kz to kt
        ktind = np.round(time_extent_mm * kt2[xind, :]) + ceil((num_time_samples - 1) / 2) + 1
        # conj().num_time_samples
        ktind = ktind.reshape(ktind.shape[0], 1) * np.ones((1, fourier_coefficients_dim)) + \
                np.ones((num_time_samples, 1)) * ktrange.reshape(ktrange.shape[0], 1).T

        ktind[ktind > num_time_samples] = ktind[ktind > num_time_samples] - num_time_samples
        ktind[ktind < 1] = ktind[ktind < 1] + num_time_samples

        # V is a helper matrix, which contains for a given kx all the kt-rated signal fourier amplitudes
        # which then will be used for interpolation of all kz-related image fourier amplitudes
        index = xind + (ktind - 1) * num_detectors
        V = sigtrans.flatten(order='F')[index.astype(int).flatten(order='F')]. \
            reshape(index.shape[0], index.shape[1], order='F')
        # Kt is a helper matrix, which contains for a given kx the kt values corresponding to all the
        # kt-related signal fourier amplitudes which will be used for interpolation
        Kt = kt.flatten(order='F')[index.astype(int).flatten(order='F')]. \
            reshape(index.shape[0], index.shape[1], order='F')

        # calculate the complex interpolation weights for interpolation of kz-related image fourier amplitudes
        # from the corresponding multiple of kt-related signal fourier amplitudes (based on paper by Jaeger et. al)
        # the distance between kt2 and Kt
        deltakt = kt2[xind, :].reshape(kt2[xind, :].shape[0], 1) * np.ones((1, fourier_coefficients_dim)) - Kt
        # prepare the coefficient matrix
        coeff = np.ones((num_time_samples, fourier_coefficients_dim))
        coeff = coeff.astype(complex)

        selection_non_zero_elements = deltakt != 0
        deltakt_coeff = deltakt.flatten(order='F')[selection_non_zero_elements.flatten(order='F')]
        deltakt_coeff_result = (1 - np.exp((-2) * pi * 1j * deltakt_coeff * time_extent_mm)) / \
                               (2 * pi * 1j * deltakt_coeff * time_extent_mm)

        coeff_FLAT = coeff.flatten(order='F')

        coeff_FLAT[selection_non_zero_elements.flatten(order='F')] = deltakt_coeff_result
        coeff = coeff_FLAT.reshape(coeff.shape[0], coeff.shape[1], order='F')
        # interpolation as a weighted sum over multiple of kt-related signal fourier amplitudes, resulting in the
        # kz-related image fourier amplitudes
        ptrans[xind, :] = np.sum(V * coeff, axis=1).T * jacobinate[xind, :]
    # only negative kt are valid
    ptrans[kt > 0] = 0

    # if the signal acquisition was started at a time different to the excitation time
    # fourier amplitudes have to be compensated for the corresponding phase
    ptrans = ptrans * np.exp((-2) * pi * 1j * kt2 * delay * sos * 10 ** (-3))
    ptrans = ptrans * np.exp(2 * pi * 1j * kz * delay * sos * 10 ** (-3))

    # inverse fourier transformation of iamge fourier amplitudes to get the image
    p = np.real(np.fft.ifft2(np.fft.ifftshift(ptrans)))

    # if zeropadding in t-direction was used, the image is cropped to the corresponding to the intial data
    if zero_pad_time:
        num_z_samples = num_z_samples / 2
        p = p[:, 0:int(num_z_samples)]

    # if zeropadding in x-direction was used, the image is cropped to the corresponding to the intial data
    if zero_pad_detectors:
        num_detectors = num_detectors - 2 * half_num_detectors
        p = p[half_num_detectors + np.arange(0, num_detectors), :]

    reconstructed_image = p.conj().T

    # rescaling (reconstructed_image is formatted: [Z, X])
    rekon_shape = np.shape(reconstructed_image)
    target_dimensions_m = np.asarray([(rekon_shape[0] * sos) /
                                      sampling_rate,
                                      rekon_shape[1] * pitch])
    #target_dimensions_m = np.asarray([(rekon_shape[0] * sos) /
                                      #sampling_rate,
                                      #field_of_view[1]-field_of_view[0]]) # zoom in according to x span
    target_dimensions_voxels = target_dimensions_m / spacing_m
    zoom_values = target_dimensions_voxels / rekon_shape
    reconstructed_image = zoom(reconstructed_image, zoom_values)

    # Check along which axis the transducer elements are aligned.
    # This code assumed that the alignment is along one axis.
    x_aligned = False
    y_aligned = False
    if pitch_x_y_z[0] > 1e-10:
        x_aligned = True
    if pitch_x_y_z[1] > 1e-10:
        y_aligned = True

    reconstructed_image = reconstructed_image.T
    reko_shape = np.shape(reconstructed_image)

    if field_of_view[0] < 0:
        field_of_view[1] = field_of_view[1] - field_of_view[0]
        field_of_view[0] = 0
    if field_of_view[2] < 0:
        field_of_view[3] = field_of_view[3] - field_of_view[2]
        field_of_view[2] = 0
    if field_of_view[4] < 0:
        field_of_view[5] = field_of_view[5] - field_of_view[4]
        field_of_view[4] = 0

    target_voxels = np.round(field_of_view / spacing_m).astype(np.int)

    if x_aligned:
        reconstructed_image = reconstructed_image.reshape((reko_shape[0], 1, reko_shape[1]))
        target_voxels[3] = 1
    elif y_aligned:
        reconstructed_image = reconstructed_image.reshape((1, reko_shape[0], reko_shape[1]))
        target_voxels[1] = 1
    else:
        reconstructed_image = reconstructed_image.reshape((reko_shape[0], reko_shape[1], 1))
        target_voxels[5] = 1

    # cropping
    reconstructed_image = reconstructed_image[target_voxels[0]:target_voxels[1],
                                              target_voxels[2]:target_voxels[3],
                                              target_voxels[4]:target_voxels[5]]

    return reconstructed_image
