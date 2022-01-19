"""
SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Mengjie Shi
SPDX-FileCopyrightText: 2022 Janek Gröhl
SPDX-License-Identifier: MIT
"""

import numpy as np
from math import *
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm
from scipy.signal import hilbert
from scipy.ndimage import zoom
from image_reconstruction.reconstruction_utils.post_processing import hilbert_transform_1_d
from image_reconstruction.reconstruction_utils.post_processing import log_compression


class FFTbasedAlgorithm(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
         Implementation of a FFT-based algorithm without any additional features.

         :param time_series_data: A 2D numpy array with the following internal array definition:
                                 [detectors, time samples]
         :param detection_elements: A dictionary that describes the detection geometry.
                                    The dictionary contains three entries:
                                    ** "positions": The positions of the detection elements relative to the field of view
                                    ** "orientations": The orientations of the detection elements
                                    ** "sizes": The sizes of the detection elements.
         :param field_of_view: A 1D 6 element-long numpy array that contains the extent of the field of view in x, y and
                               z direction in the same coordinate system as the detection element positions.


         :param kwargs: the list of parameters for the fourier domain reconstruction includes the following parameters:
             ** 'speed_of_sound_m_s' the target speed of sound in units of meters per second
             ** 'delay': time delay from laser irradiation to signal acquisition start (default 0)
             ** 'zero_pad_detectors': 1=zero pad in lateral (X) direction; 0=don't typically 1
             ** 'zero_pad_time': 1=zero pad in axial (t,time) direction; 0=don't typically 1
             ** 'coeffT': signal fourier coefficients a single image fourier coefficient is interploated (5)
             ** 'samplingX': 1,defines how many image lines are reconstructed per transducer element. For value>1, the additional image lines are equidistantly placed between the transducer elements
             ** 'target_resolution': the target resolution in meters. Default resolution is 0.1 mm.

         :return:
         """
        speed_of_sound_in_m_per_s = 1480
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]

        pitch_x_y_z = np.abs(detection_elements["positions"][1] - detection_elements["positions"][0])
        element_pitch = np.max(pitch_x_y_z)
        # Caution! This assumed that we have a linear uniform progression of
        # the elements along one axis only!

        delay = 0
        if "delay" in kwargs:
            delay = kwargs["delay"]

        zero_pad_detectors = 1
        if "zero_pad_detectors" in kwargs:
            zero_pad_detectors = kwargs["zero_pad_detectors"]

        zero_pad_time = 1
        if "zero_pad_time" in kwargs:
            zero_pad_time = kwargs["zero_pad_time"]

        coeffT = 5
        if "coeffT" in kwargs:
            coeffT = kwargs["coeffT"]

        samplingX = 1
        if "samplingX" in kwargs:
            samplingX = kwargs["samplingX"]

        target_resolution = 0.0001
        if "target_resolution" in kwargs:
            target_resolution = kwargs["target_resolution"]

        envelope = False
        if "envelope" in kwargs:
            envelope = kwargs["envelope"]

        envelope_type = None
        if "envelope_type" in kwargs:
            envelope_type = kwargs["envelope_type"]

        rekon, rekonuncut = self.rekon_OA_freqdom(time_series_data,
                                                  pitch=element_pitch,
                                                  sampling_rate=self.ipasc_data.get_sampling_rate(),
                                                  sos=speed_of_sound_in_m_per_s,
                                                  delay=delay,
                                                  zero_pad_detectors=zero_pad_detectors,
                                                  zero_pad_time=zero_pad_time,
                                                  coeffT=coeffT,
                                                  samplingX=samplingX)

        # rescaling (rekon is formatted: [Z, X])
        rekon_shape = np.shape(rekon)
        target_dimensions_m = np.asarray([(rekon_shape[0] * speed_of_sound_in_m_per_s) /
                                          self.ipasc_data.get_sampling_rate(),
                                          rekon_shape[1] * element_pitch])
        target_dimensions_voxels = target_dimensions_m / target_resolution
        zoom_values = target_dimensions_voxels / rekon_shape

        rekon = zoom(rekon, zoom_values)

        # Check along which axis the transducer elements are aligned.
        # This code assumed that the alignment is along one axis.
        x_aligned = False
        y_aligned = False
        if pitch_x_y_z[0] > 1e-10:
            x_aligned = True
        if pitch_x_y_z[1] > 1e-10:
            y_aligned = True

        rekon = rekon.T
        reko_shape = np.shape(rekon)
        target_voxels = (field_of_view / target_resolution).astype(np.int)
        if x_aligned:
            rekon = rekon.reshape((reko_shape[0], 1, reko_shape[1]))
            target_voxels[3] = 1
        elif y_aligned:
            rekon = rekon.reshape((1, reko_shape[0], reko_shape[1]))
            target_voxels[1] = 1
        else:
            rekon = rekon.reshape((reko_shape[0], reko_shape[1], 1))
            target_voxels[5] = 1

        # cropping
        rekon = rekon[target_voxels[0]:target_voxels[1],
                target_voxels[2]:target_voxels[3],
                target_voxels[4]:target_voxels[5]]

        if envelope:
            if envelope_type == "hilbert":
                # hilbert transform
                rekon = hilbert_transform_1_d(rekon, axis=0)
            elif envelope_type == "log":
                # hilbert transform + log-compression on 40 dB
                rekon = log_compression(rekon, axis=0, dynamic=40)
            elif envelope_type == "zero":
                # zero forcing
                rekon[rekon < 0] = 0
            elif envelope_type == "abs":
                # absolute value
                rekon = np.abs(rekon)
            else:
                print("WARN: No envelope type specified!")

        return rekon

    @staticmethod
    def rekon_OA_freqdom(time_series_data,
                         pitch: float,
                         sampling_rate,
                         sos: float,
                         delay,
                         zero_pad_detectors,
                         zero_pad_time,
                         coeffT,
                         samplingX):

        """

        :param time_series_data: A 2D numpy array with the following internal array definition:
                                [detectors, time samples]
        :param pitch: transducer element pitch in units of meters (AcousticX 0.315 mm)
        :param sampling_rate: data sampling rate (AcousticX 40MHz)
        :param sos: speed of sound in units of meters
        :param delay: time delay from laser irradiation to signal acquisition start (default 0)
        :param zero_pad_detectors: 1=zero pad in lateral (X) direction; 0=don't typically 1
        :param zero_pad_time: 1=zero pad in axial (t,time) direction; 0=don't typically 1
        :param coeffT: signal fourier coefficients a single image fourier coefficient is interploated (5)
        :param samplingX: this defines how many image lines are reconstructed per transducer element.
        For value>1, the additional image lines are equidistantly placed between the transducer elements


        :return: **rekon: resulting image, has equal number of samples in axial(z) direction as the signal.
                 The number of image lines is: (number of elements -1) *samplingX +1

                 **rekonuncut: if the zeroX flag was set, then this variable contains the full image corresponding to
                 the full (virtual) size of the padded aperture, while rekon contains only the part of the image
                 correspond to the real aperture.
        """

        # DATA DIMENSIONS
        # dimension of the signal and image in number of samples
        num_detectors = time_series_data.shape[0] # check X is 128
        num_z_samples = time_series_data.shape[1]
        num_time_samples = num_z_samples
        # corresponding physical dimension of signal and image in [mm]
        x_extent = num_detectors * pitch * 10**3
        Zextent = num_z_samples * sos * 10**(-3) / (sampling_rate * 10 ** (-6)) # check if the rate in Mhz
        Textent = num_time_samples * sos * 10**(-3) / (sampling_rate * 10 ** (-6))

        # DATA PRE-CONDITIONING
        # lateral oversampling of the image, compared to real element pitch
        # this measure leads to finer images.
        # for this purpose, zero signals are inserted between real signals
        # and the element pitch is correspondingly reduced.
        if samplingX > 1: # need check with sampling > 1
            # generate a zero array with the appropriate size
            sig2 = np.zeros(samplingX * (num_detectors-1) + 1, num_z_samples) # match the type of time_series_data?
            # signal lines corresponds to real elements are filled
            sig2[0:(num_detectors-1)*samplingX+1, :] = time_series_data
            # update the pitch size
            pitch = pitch / samplingX  # check
            # update the number of samples
            num_detectors = samplingX * (num_detectors-1) +1
            # the aperture is adapted
            x_extent = num_detectors * pitch
            time_series_data = sig2

        # zero padding of the signal data in num_time_samples-direction, by a factor of two
        # this measure reduces aliasing artifacts when a strong OA source is located either
        # close to the start or to the end of the image frame
        # generally the influence is quite low, if coeffT is set to a large value such as 5.
        if zero_pad_time:
            # the signal matrix is padded to double num_time_samples-size
            time_series_data = np.append(time_series_data, np.zeros((num_detectors, num_time_samples)), axis=1)
            # dimensions of image frame are doubled
            num_z_samples = 2 * num_z_samples
            Zextent = 2 * Zextent
            # dimensions of signal frame are doubled
            num_time_samples = 2 * num_time_samples
            Textent = 2 * Textent

        # zero padding of the signal data in X-direction, by a factor of two
        # this measure reduces aliasing artifacts when a strong OA source is located close
        # to either side of the image frame, if a source is located outside the image frame,
        # it may be correctly reconstructed, leading to less artifacts.
        if zero_pad_detectors:
            half_num_detectors = round(num_detectors/2) # check if round to the nearest interger (floating)
            delta_x_extent = x_extent/2
            # dimensions of image field are extended
            time_series_data = np.concatenate((np.zeros((half_num_detectors, num_time_samples)),
                                               time_series_data,
                                               np.zeros((half_num_detectors, num_time_samples))), axis=0)
            num_detectors = num_detectors + 2 * half_num_detectors
            x_extent = x_extent + 2 * delta_x_extent

        # RECONSTRUCTION ALGORITHM
        # 2D array containing kx and kz values of k-vectors of the image fourier space
        [k_x, k_z] = np.meshgrid(np.arange(-ceil((num_detectors-1)/2), floor((num_detectors-1)/2)+1)/x_extent, np.arange(-ceil((num_z_samples-1)/2), floor((num_z_samples-1)/2)+1)/Zextent)
        kz = k_z.T
        kx = k_x.T
        # 2D array containing kt values of k-vectors of the signal fourier space
        kt = kz # array(2046, 256)
        # 1D array containing values of kt
        kate = np.arange(-ceil((num_z_samples-1)/2), floor((num_z_samples-1)/2)+1)/Zextent
        # maximum value of kt
        katemax = ceil((num_z_samples-1)/2)/Zextent

        # projection of (kx,kz) onto (kx,kt) describing the signal formation in the fourier space
        # projection of kz onto kt

        kt2 = -np.sqrt(kz**2+kx**2) # check math.sqrt， power of each elememt
        # calculate the jacobinate of the projection of kz onto kt

        kt2[(kz == 0) & (kx == 0)] = 1 # &
        jakobiante = kz / kt2
        jakobiante[(kz == 0) & (kx == 0)] = 1
        kt2[(kz == 0) & (kx == 0)] = 0

        # if kt2 out of bound of kt, take the modulo value
        samplfreq = num_time_samples/Textent # num_time_samples is 1024 (without sampling)
        kt2 = (kt2 + samplfreq/2) % samplfreq - samplfreq/2

        # 2D fourier spectrum of signal data
        sigtrans = np.fft.fft2(time_series_data)
        sigtrans = np.fft.fftshift(sigtrans)
        #only negatively propagating waves are detected,
        sigtrans[kz > 0] = 0

        # initialize array for image fourier coefficients
        ptrans = np.zeros((num_detectors, num_z_samples)).astype(complex)

        # numbers of signal fourier coefficients used for interpolation of image
        # fourier coefficient, in up and down direction
        nTup = ceil((coeffT - 1)/2)
        nTdo = floor((coeffT -1)/2)

        # helper index vector covering the range of fourier amplitudes going into interpolation
        ktrange = np.arange(-nTdo, nTup+1) # vector (5,)

        # loop for interpolation of image fourier coefficients from signal fourier coefficients
        # for each image line do
        for xind in range(0, num_detectors):
            # calculate kt-index into kt-dimension of signal fourier space, from the kt value resulting
            # from the projection of kz to kt
            ktind = np.round(Textent * kt2[xind, :]) + ceil((num_time_samples-1)/2) + 1
            # conj().num_time_samples
            ktind = ktind.reshape(ktind.shape[0],1) * np.ones((1, coeffT)) + np.ones((num_time_samples, 1)) * ktrange.reshape(ktrange.shape[0], 1).T

            ktind[ktind > num_time_samples] = ktind[ktind > num_time_samples] - num_time_samples
            ktind[ktind < 1] = ktind[ktind < 1] + num_time_samples

            # V is a helper matrix, which contains for a given kx all the kt-rated signal fourier amplitudes
            # which then will be used for interpolation of all kz-related image fourier amplitudes
            index = xind  + (ktind - 1) * num_detectors
            V = sigtrans.flatten(order='F')[index.astype(int).flatten(order='F')].reshape(index.shape[0], index.shape[1], order='F')
            # Kt is a helper matrix, which contains for a given kx the kt values corresponding to all the
            # kt-related signal fourier amplitudes which will be used for interpolation
            Kt = kt.flatten(order='F')[index.astype(int).flatten(order='F')].reshape(index.shape[0], index.shape[1], order='F')

            # calculate the complex interpolation weights for interpolation of kz-related image fourier amplitudes
            # from the corresponding multiple of kt-related signal fourier amplitudes (based on paper by Jaeger et. al)
            # the distance between kt2 and Kt
            deltakt = kt2[xind, :].reshape(kt2[xind,:].shape[0], 1) * np.ones((1, coeffT)) - Kt
            # prepare the coefficient matrix
            coeff = np.ones((num_time_samples, coeffT))
            coeff = coeff.astype(complex)

            help = deltakt!=0 #
            deltakt_coeff = deltakt.flatten(order='F')[help.flatten(order='F')]
            deltakt_coeff_result = (1 - np.exp((-2)*pi*1j*deltakt_coeff*Textent))/(2*pi*1j*deltakt_coeff*Textent)

            coeff_FLAT = coeff.flatten(order='F')

            coeff_FLAT[help.flatten(order='F')] = deltakt_coeff_result
            coeff = coeff_FLAT.reshape(coeff.shape[0], coeff.shape[1], order='F')
            # interpolation as a weighted sum over multiple of kt-related signal fourier amplitudes, resulting in the
            # kz-related image fourier amplitudes
            ptrans[xind, :] = np.sum(V*coeff, axis=1).T * jakobiante[xind, :]
        # only negative kt are valid
        ptrans[kt>0] = 0

        # if the signal acquisition was started at a time different to the excitation time
        # fourier amplitudes have to be compensated for the corresponding phase
        ptrans = ptrans * np.exp((-2) * pi * 1j * kt2 * delay * sos * 10**(-3))
        ptrans = ptrans * np.exp(2 * pi * 1j * kz * delay * sos * 10**(-3))

        # inverse fourier transformation of iamge fourier amplitudes to get the image
        p = np.real(np.fft.ifft2(np.fft.ifftshift(ptrans)))

        # if zeropadding in t-direction was used, the image is cropped to the corresponding to the intial data
        if zero_pad_time:
            num_z_samples = num_z_samples/2
            Zextent = Zextent/2
            p = p[:, 0:int(num_z_samples)]
            num_time_samples = num_time_samples/2
            Textent = Textent/2
        # if zeropadding in x-direction was used, the image is cropped to the corresponding to the intial data
        puncut = p
        if zero_pad_detectors:
            num_detectors = num_detectors - 2*half_num_detectors
            p = p[half_num_detectors + np.arange(0, num_detectors),:]
        # if lateral subsampling was used, the resulting image must be lateral filtered in order to
        # reduce undersampling artifacts
        def conditioner(signal, rad, downsampl, laser):

            rad2 = np.sqrt(rad ** 2 + laser ** 2)
            blur = cos((2 * pi * np.arange((-2) * rad, 2 * rad + 1)).conj().T / 4 * rad) + 1
            blur = blur / np.sum(blur)

            signal2 = np.convolve(signal, blur, model='same')  # not sure if equivelent to convn
            signal2 = np.reshape(signal2, [1, downsampl])

            return signal2

        if samplingX>1:
            p = conditioner(p, samplingX/2, 1, 0)
            puncut = conditioner(puncut, 1, 1, 0)

        rekon = p.conj().T
        rekonuncut = puncut.conj().T

        rekon = abs(hilbert(rekon, axis=-2))
        rekon = rekon**2

        return rekon, rekonuncut
