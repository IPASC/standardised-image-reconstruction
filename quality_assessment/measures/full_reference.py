"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-License-Identifier: MIT
"""

import numpy as np
from quality_assessment.measures import FullReferenceMeasure
from sewar import ssim, uqi, rmse
from sklearn.metrics import mutual_info_score


class StructuralSimilarityIndex(FullReferenceMeasure):

    def compute_measure(self, ground_truth_image, reconstructed_image):
        return ssim(ground_truth_image, reconstructed_image, ws=8, MAX=np.max([np.max(ground_truth_image), np.max(reconstructed_image)]))[0]

    def get_name(self):
        return "SSIM"


class UniversalQualityIndex(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return uqi(expected_result, reconstructed_image, ws=8)

    def get_name(self):
        return "UQI"


class RootMeanSquaredError(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return rmse(expected_result, reconstructed_image)

    def get_name(self):
        return "RMSE"


class MutualInformation(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return mutual_info_score(expected_result.reshape((-1, )), reconstructed_image.reshape((-1, )))

    def get_name(self):
        return "MI"
