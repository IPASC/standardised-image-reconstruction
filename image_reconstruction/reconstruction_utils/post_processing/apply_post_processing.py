
import numpy as np
from image_reconstruction.reconstruction_utils.post_processing.envelope_detection import hilbert_transform_1_d


def apply_post_processing(reconstruction, **kwargs):

    envelope = False
    if "envelope" in kwargs:
        envelope = kwargs["envelope"]

    envelope_type = None
    if "envelope_type" in kwargs:
        envelope_type = kwargs["envelope_type"]

    if envelope:
        if envelope_type == "hilbert":
            # hilbert transform
            reconstruction = hilbert_transform_1_d(reconstruction, axis=0)
        elif envelope_type == "hilbert_squared":
            # hilbert transform + squaring
            reconstruction = hilbert_transform_1_d(reconstruction, axis=0)
            reconstruction = reconstruction**2
        elif envelope_type == "log":
            # hilbert transform + log-compression
            reconstruction = hilbert_transform_1_d(reconstruction, axis=0)
            # do 20log10 on the normalized image
            reconstruction = 20 * np.log10(reconstruction / np.nanmax(reconstruction))
        elif envelope_type == "log_squared":
            # hilbert transform + squaring + log-compression
            reconstruction = hilbert_transform_1_d(reconstruction, axis=0)
            reconstruction = reconstruction ** 2
            # do 20log10 on the normalized image
            reconstruction = 20 * np.log10(reconstruction / np.nanmax(reconstruction))
        elif envelope_type == "zero":
            # zero forcing
            reconstruction[reconstruction < 0] = 0
        elif envelope_type == "abs":
            # absolute value
            reconstruction = np.abs(reconstruction)
        else:
            print(f"WARN: No valid envelope type specified! Was: {envelope_type}")

    return reconstruction
