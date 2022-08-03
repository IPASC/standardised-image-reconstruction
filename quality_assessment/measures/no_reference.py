from quality_assessment.measures import NoReferenceMeasure
import numpy as np


class GeneralisedSignalToNoiseRatio(NoReferenceMeasure):

    def compute_measure(self, reconstructed_image, signal_roi, noise_roi, n_bins=256):
        """
        Implemented from TODO CITE
        using the histogram-based approximation

        Parameters
        ----------
        reconstructed_image
        signal_roi
        noise_roi
        n_bins

        Returns
        -------

        """

        signal_min = np.nanmin(reconstructed_image)
        signal_max = np.nanmax(reconstructed_image)
        reconstructed_image = (reconstructed_image - signal_min) / (signal_max - signal_min)
        reconstructed_image = reconstructed_image * 256

        value_range = np.arange(0, 257)
        signal_hist = np.histogram(reconstructed_image[signal_roi], bins=value_range,
                                   density=True)[0]
        noise_hist = np.histogram(reconstructed_image[noise_roi], bins=value_range,
                                  density=True)[0]
        overlap = 0
        for i in range(n_bins):
            overlap = overlap + np.min([signal_hist[i], noise_hist[i]])

        return 1 - overlap