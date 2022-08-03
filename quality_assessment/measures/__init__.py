import abc
from abc import ABC


class FullReferenceMeasure(ABC):

    @abc.abstractmethod
    def compute_measure(self, expected_result, reconstructed_image):
        pass


class NoReferenceMeasure(ABC):

    @abc.abstractmethod
    def compute_measure(self, reconstructed_image, signal_roi, noise_roi):
        pass