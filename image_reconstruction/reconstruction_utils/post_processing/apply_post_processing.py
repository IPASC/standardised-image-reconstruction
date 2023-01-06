
import numpy as np
from image_reconstruction.reconstruction_utils.post_processing.envelope_detection import hilbert_transform_1_d


def apply_post_processing(recon, **kwargs):

    non_negativity_method = None
    if "non_negativity_method" in kwargs:
        non_negativity_method = kwargs["non_negativity_method"]

    if non_negativity_method is not None:
        if non_negativity_method == "hilbert":
            # hilbert transform
            recon = hilbert_transform_1_d(recon, axis=0)
        elif non_negativity_method == "hilbert_squared":
            # hilbert transform + squaring
            recon = hilbert_transform_1_d(recon, axis=0)
            recon = recon ** 2
        elif non_negativity_method == "log":
            # hilbert transform + log-compression
            recon = hilbert_transform_1_d(recon, axis=0)
            # do 20log10 on the normalized image
            recon = 20 * np.log10(recon / np.nanmax(recon))
        elif non_negativity_method == "log_squared":
            # hilbert transform + squaring + log-compression
            recon = hilbert_transform_1_d(recon, axis=0)
            recon = recon ** 2
            # do 20log10 on the normalized image
            recon = 20 * np.log10(recon / np.nanmax(recon))
        elif non_negativity_method == "zero":
            # zero forcing
            recon[recon < 0] = 0
        elif non_negativity_method == "abs":
            # absolute value
            recon = np.abs(recon)
        else:
            print(f"WARN: No valid envelope type specified! Was: {non_negativity_method}")

    return recon
