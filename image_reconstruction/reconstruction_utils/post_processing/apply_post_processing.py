
import numpy as np
from image_reconstruction.reconstruction_utils.post_processing.envelope_detection import hilbert_transform_1_d
from image_reconstruction.reconstruction_utils.post_processing.envelope_detection import log_compression


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
        elif envelope_type == "log":
            # hilbert transform + log-compression
            reconstruction = log_compression(reconstruction, axis=0)
        elif envelope_type == "zero":
            # zero forcing
            reconstruction[reconstruction < 0] = 0
        elif envelope_type == "abs":
            # absolute value
            reconstruction = np.abs(reconstruction)
        else:
            print("WARN: No envelope type specified!")

    return reconstruction
