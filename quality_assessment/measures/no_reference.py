"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-License-Identifier: MIT
"""

from quality_assessment.measures import NoReferenceMeasure
import numpy as np


class GeneralisedSignalToNoiseRatio(NoReferenceMeasure):

    def compute_measure(self, reconstructed_image, signal_roi, noise_roi):
        """
        Implemented from the paper by Kempski et al 2020::

            Kempski, Kelley M., et al.
            "Application of the generalized contrast-to-noise ratio to assess photoacoustic image quality."
            Biomedical Optics Express 11.7 (2020): 3684-3698.

        This implementation uses the histogram-based approximation.

        Parameters
        ----------
        reconstructed_image
        signal_roi
            must be in the same shape as the reconstructed image
        noise_roi
            must be in the same shape as the reconstructed image

        Returns
        -------
        float, a measure of the relative overlap of the signal probability densities.

        """

        # rescale signal into value range of 0 - 256 to mimic the original paper bin sizes
        signal_min = np.nanmin(reconstructed_image)
        signal_max = np.nanmax(reconstructed_image)
        reconstructed_image = (reconstructed_image - signal_min) / (signal_max - signal_min)
        reconstructed_image = reconstructed_image * 256

        # define 256 unit size bins (257 bin edges) and compute the histogram PDFs.
        value_range_bin_edges = np.arange(0, 257)
        signal_hist = np.histogram(reconstructed_image[signal_roi], bins=value_range_bin_edges,
                                   density=True)[0]
        noise_hist = np.histogram(reconstructed_image[noise_roi], bins=value_range_bin_edges,
                                  density=True)[0]

        # compute the overlap
        overlap = 0
        for i in range(256):
            overlap = overlap + np.min([signal_hist[i], noise_hist[i]])

        # return gCNR
        return 1 - overlap

    def get_name(self):
        return "gCNR"
