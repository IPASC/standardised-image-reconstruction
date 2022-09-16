"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-FileCopyrightText: 2022 Lina Hacker
SPDX-License-Identifier: MIT
"""

import abc
from abc import ABC


class PerformanceMeasure(ABC):

    @abc.abstractmethod
    def get_name(self):
        pass


class FullReferenceMeasure(PerformanceMeasure):

    @abc.abstractmethod
    def compute_measure(self, expected_result, reconstructed_image):
        pass


class NoReferenceMeasure(PerformanceMeasure):

    @abc.abstractmethod
    def compute_measure(self, reconstructed_image, signal_roi, noise_roi):
        pass