"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-FileCopyrightText: 2022 Shufan Yang
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
from sewar import rmse
from sklearn.metrics import mutual_info_score
from quality_assessment.measures import FullReferenceMeasure
from torchmetrics import StructuralSimilarityIndexMeasure, UniversalImageQualityIndex
from scipy.stats import linregress
from scipy.signal import correlate
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine


def validate_input(a, b):
    if len(a) != len(b):
        raise AssertionError("The number of wavelengths must be the same for both input samples")


def get_torch_tensor(np_array):
    """
    Takes a 2D or 3D numpy array representing a greyscale image and transforms it into a torch sensor for the
    purposes of computing an image quality measure.

    :param np_array: a 2D or 3D numpy array
    :return: a torch tensor of shape (1, zdim, xdim, ydim)
    """
    shape = np_array.shape
    if len(shape) == 2:
        sx, sy = shape
        sz = 1
    elif len(shape) == 3:
        sx, sy, sz = shape
    else:
        raise AssertionError("The input image must be 2D or 3D")

    return torch.from_numpy(np_array.reshape((1, sz, sx, sy)))


class StructuralSimilarityIndex(FullReferenceMeasure):
    def compute_measure(self, expected_result, reconstructed_image):
        gt = get_torch_tensor(expected_result)
        reco = get_torch_tensor(reconstructed_image)
        return 1 - StructuralSimilarityIndexMeasure()(gt, reco).item()

    def get_name(self):
        return "Structural Similarity Index"


class UniversalQualityIndex(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        gt = get_torch_tensor(expected_result)
        reco = get_torch_tensor(reconstructed_image)
        return 1 - UniversalImageQualityIndex()(gt, reco).item()

    def get_name(self):
        return "Universal Quality Index"


class RootMeanSquaredError(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return rmse(expected_result, reconstructed_image)

    def get_name(self):
        return "RMSE"


class MutualInformation(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        gt = self.precompute(expected_result)
        reco = self.precompute(reconstructed_image)
        res = mutual_info_score(gt, reco)
        if res == 0:
            return 1
        return 1 / res

    def precompute(self, data):
        data = data.reshape((-1, ))
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data * 256
        return data.astype(int)

    def get_name(self):
        return "Mutual Information"


class WassersteinDistance(FullReferenceMeasure):
    def compute_measure(self, expected_result, reconstructed_image):
        return self.compute_wasserstein_distance(expected_result, reconstructed_image)

    @staticmethod
    def compute_wasserstein_distance(data1, data2):
        validate_input(data1, data2)

        emd = 0
        for i in range(len(data1)):
            emd += wasserstein_distance(data1[i], data2[i])
        emd = emd / len(data1)
        return emd

    def get_name(self):
        return "Wasserstein Distance"


class CosineDistance(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return self.compute_cosine_distance(np.reshape(expected_result, (1, -1)),
                                            np.reshape(reconstructed_image, (1, -1)))

    @staticmethod
    def compute_cosine_distance(data1, data2):
        validate_input(data1, data2)

        cos_distance = 0
        for i in range(len(data1)):
            cos_distance += cosine(data1[i], data2[i])
        cos_distance = cos_distance / len(data1)
        return cos_distance

    def get_name(self):
        return "Cosine Distance"


class NormalisedCrossCorrelation(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        res = self.compute_normalized_cross_correlation(np.reshape(expected_result, (1, -1)),
                                                        np.reshape(reconstructed_image, (1, -1)))
        if res == 0:
            return 1
        else:
            return 1/res

    @staticmethod
    def compute_normalized_cross_correlation(data1, data2):
        # Ensure both input images are of the same size
        validate_input(data1, data2)

        ncc = 0
        for i in range(len(data1)):
            a = data1[i]
            b = data2[i]

            # Compute the mean of each image
            mean1 = np.mean(a)
            mean2 = np.mean(b)

            # Compute the cross-correlation using correlate2d
            cross_correlation = correlate(a - mean1, b - mean2, mode='full')

            # Compute the standard deviations of both images
            std1 = np.std(a)
            std2 = np.std(b)

            # Compute the NCC
            ncc += cross_correlation / (std1 * std2 * a.size)

        ncc = ncc / len(data1)

        return np.mean(ncc)

    def get_name(self):
        return "Normalised Cross Correlation"


class BhattacharyyaDistance(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return self.compute_bhattacharyya_distance(np.reshape(expected_result, (1, -1)),
                                                   np.reshape(reconstructed_image, (1, -1)))

    @staticmethod
    def compute_bhattacharyya_distance(data1, data2, range_min=-3, range_max=3, num_bins=61):

        bd = 0
        for i in range(len(data1)):
            a = data1[i]
            b = data2[i]
            # Z-score normalization
            mean1, std1 = np.mean(a), np.std(a)
            mean2, std2 = np.mean(b), np.std(b)

            zscore_normalized_image1 = (a - mean1) / std1
            zscore_normalized_image2 = (b - mean2) / std2

            # Create histograms
            hist1, bin_edges1 = np.histogram(zscore_normalized_image1, bins=num_bins, range=(range_min, range_max))
            hist2, bin_edges2 = np.histogram(zscore_normalized_image2, bins=num_bins, range=(range_min, range_max))

            # Calculate the Bhattacharyya distance
            hist1 = hist1 / np.sum(hist1)  # Normalize histograms
            hist2 = hist2 / np.sum(hist2)

            bd += -np.log(np.sum(np.sqrt(hist1 * hist2)))

        bd = bd / len(data1)

        return bd

    def get_name(self):
        return "Bhattacharyya Distance"


class KullbackLeiblerDivergence(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return self.compute_kullback_leibler_divergence(np.reshape(expected_result, (1, -1)),
                                                        np.reshape(reconstructed_image, (1, -1)))

    @staticmethod
    def compute_kullback_leibler_divergence(a, b):
        validate_input(a, b)
        # Normalise the data
        a = (a - np.mean(a)) / np.std(a)
        b = (b - np.mean(b)) / np.std(b)

        # Compute discrete KLD from marginal histograms
        kld = 0
        for wl_idx in range(len(a)):
            marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
            marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
            marginal_p = marginal_p + 0.00001
            marginal_q = marginal_q + 0.00001
            kld += entropy(marginal_p, marginal_q, base=2)
        kld = kld / len(a)
        return kld

    def get_name(self):
        return "Kullback-Leibler Divergence"


class JensenShannonDivergence(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return self.compute_jensen_shannon_divergence(np.reshape(expected_result, (1, -1)),
                                                      np.reshape(reconstructed_image, (1, -1)))

    @staticmethod
    def compute_jensen_shannon_divergence(a, b):
        validate_input(a, b)
        # Normalise the data
        a = (a - np.mean(a)) / np.std(a)
        b = (b - np.mean(b)) / np.std(b)

        # Compute discrete JSD from marginal histograms
        jsd = 0
        for wl_idx in range(len(a)):
            marginal_p, _ = np.histogram(a[wl_idx], bins=np.arange(-3, 3, 6 / 100))
            marginal_q, _ = np.histogram(b[wl_idx], bins=np.arange(-3, 3, 6 / 100))
            marginal_p = marginal_p + 0.00001
            marginal_q = marginal_q + 0.00001
            jsd += jensenshannon(marginal_p, marginal_q, base=2)

        jsd = jsd / len(a)
        return jsd

    def get_name(self):
        return "Jensen Shannon Divergence"
