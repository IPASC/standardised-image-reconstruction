from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
import matplotlib.pyplot as plt

# #####################################################################
# TEST IMAGES DOCUMENTATION
# All images are distributed via the MIT license
# #####################################################################
# #####################################################################
#
# IMAGE_IDX = 0
# Simulated image of two tubular structured underneath a horizontal layer
# provided by Janek Gröhl.
#
# #####################################################################
#
# IMAGE_IDX = 1
# Simulated image of point sources in a homogeneous medium provided by
# Janek Gröhl.
#
# #####################################################################
#
# IMAGE_IDX = 2
# Experimental image provided by Manojit Pramanik. It is a point absorber
# in a homogeneous medium. SOS=1480
#
# #####################################################################
#
IMAGE_IDX = 3
# Simulated image of point sources in a homogeneous medium provided by
# François Varray. 10 point absorbers are located in a homogeneous medium
# at depths between 10 and 40 mm. With increasing depth, they are
# also positioned laterally between 0 and 30 mm.
#
# #####################################################################
#
# IMAGE_IDX = 4
# Experimental measurement of a point source in a homogeneous medium.
# Measurement is provided by Mengjie Shi.
#
# #####################################################################
#
# IMAGE_IDX = 5
# Experimental measurement of a point source in a homogeneous medium.
# Measurement is provided by Mengjie Shi.
#
# #####################################################################

SPEED_OF_SOUND = 1540

out = TestDelayAndSum()
out.p_factor = 1
out.fnumber = 2.5
result1 = out.test_vanilla_delay_and_sum_reconstruction_is_running_through(IMAGE_IDX, visualise=False,
                                                                           speed_of_sound=SPEED_OF_SOUND)
result2 = out.test_delay_and_sum_reconstruction_bandpass_is_running_through(IMAGE_IDX, visualise=False,
                                                                            speed_of_sound=SPEED_OF_SOUND)
result3 = out.test_delay_and_sum_reconstruction_bandpass_pre_envelope_is_running_through(IMAGE_IDX, visualise=False,
                                                                                     speed_of_sound=SPEED_OF_SOUND)
result4 = out.test_delay_and_sum_reconstruction_bandpass_post_envelope_is_running_through(IMAGE_IDX, visualise=False,
                                                                                     speed_of_sound=SPEED_OF_SOUND)

plt.figure(figsize=(16, 4))
plt.suptitle("Various DAS reconstructions")
plt.subplot(1, 4, 1)
plt.title("Vanilla")
plt.imshow(result1[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(1, 4, 2)
plt.title("BP")
plt.imshow(result2[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(1, 4, 3)
plt.title("BP + Hilbert on p(t)")
plt.imshow(result3[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(1, 4, 4)
plt.title("BP + Hilbert on p0")
plt.imshow(result4[:, 0, :, 0, 0].T)
plt.colorbar()

plt.tight_layout()
plt.show()

# Only uncomment the following two lines if a new reference reconstruction should be saved!
# import numpy as np
# np.savez("NUMBER_reference.npz",
#          reconstruction=result1[:, 0, :, 0, 0])
