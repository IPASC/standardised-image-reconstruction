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


class StructuralSimilarityIndexTorch(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        gt = get_torch_tensor(expected_result)
        reco = get_torch_tensor(reconstructed_image)
        return StructuralSimilarityIndexMeasure()(gt, reco).item()

    def get_name(self):
        return "SSIM"


class UniversalQualityIndex(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        gt = get_torch_tensor(expected_result)
        reco = get_torch_tensor(reconstructed_image)
        return UniversalImageQualityIndex()(gt, reco).item()

    def get_name(self):
        return "UQI"


class RootMeanSquaredError(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return rmse(expected_result, reconstructed_image)

    def get_name(self):
        return "RMSE"


class MutualInformation(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        gt = self.precompute(expected_result)
        reco = self.precompute(reconstructed_image)
        return mutual_info_score(gt, reco)

    def precompute(self, data):
        data = data.reshape((-1, ))
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data * 256
        return data.astype(int)

    def get_name(self):
        return "MI"
