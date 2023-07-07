"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-FileCopyrightText: 2022 Shufan Yang
SPDX-License-Identifier: MIT
"""

import torchvision.transforms as transforms
from quality_assessment.measures import FullReferenceMeasure
from sewar import rmse
from sklearn.metrics import mutual_info_score
from torchmetrics import UniversalImageQualityIndex
from torchmetrics import StructuralSimilarityIndexMeasure



class StructuralSimilarityIndex(FullReferenceMeasure):

    def compute_measure(self, ground_truth_image, reconstructed_image):

        transform = transforms.Compose([transforms.PILToTensor()])
        reconstructed_tensor = transform(reconstructed_image)
        ground_tensor = transform(ground_truth_image)
        ssimtorch = StructuralSimilarityIndexMeasure()
        return ssimtorch(reconstructed_tensor, ground_tensor)

    def get_name(self):
        return "SSIM"

class UniversalQualityIndex(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        transform = transforms.Compose([transforms.PILToTensor()])
        reconstructed_tensor = transform(reconstructed_image)
        ground_tensor = transform(expected_result)
        return UniversalImageQualityIndex()(reconstructed_tensor, ground_tensor)

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
