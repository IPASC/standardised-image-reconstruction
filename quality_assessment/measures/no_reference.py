"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-License-Identifier: MIT
"""

#from quality_assessment.measures import NoReferenceMeasure
import numpy as np

def get_first_value(vec, val):
    for i in range(0,len(vec)):
        if (vec[i]<val):
            break
    return i


#class GeneralisedSignalToNoiseRatio(NoReferenceMeasure):
def compute_FWHM(reconstructed_image, spacing, NON_NEGATIVITY_METHOD="log", roi=[]):
    # Create the ROI is not existing
    if len(roi)==0:
        roi = 1 + 0*reconstructed_image #np.array(np.shape(reconstructed_image))
        roi = roi>0.5

    # reduce the data to the ROI
    data = reconstructed_image

    #suppress the nan
    data[np.isnan(data)] = np.nanmin(data)

    # Extract the maximum inside the roi
    ind = np.unravel_index(np.argmax(data, axis=None), data.shape)

    profil_axial = data[ind[0],:]
    profil_lateral = data[:,ind[1]]

    if (NON_NEGATIVITY_METHOD=="log"):
        i1 = get_first_value(np.flip(profil_axial[0:ind[1]+1]), -6)
        i2 = get_first_value(profil_axial[ind[1]:-1], -6)
        axial_resolution = (i2+i1) * spacing

        i1 = get_first_value(np.flip(profil_lateral[0:ind[0]+1]), -6)
        i2 = get_first_value(profil_lateral[ind[0]:-1], -6)
        lateral_resolution = (i2+i1) * spacing

        return [lateral_resolution, axial_resolution]
    else:
        print("FWHM is not implemented for other input that log images")

def compute_gCNR(reconstructed_image, signal_roi=[], noise_roi=[]):
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

#    def get_name(self):
#        return "gCNR"
