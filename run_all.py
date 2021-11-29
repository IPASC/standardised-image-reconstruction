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
# in a homogeneous medium.
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

out = TestDelayAndSum()
out.p_factor = 1
out.fnumber = 2.5
result1 = out.test_vanilla_delay_and_sum_reconstruction_is_running_through(IMAGE_IDX, visualise=False)
result2 = out.test_delay_and_sum_reconstruction_bandpass_is_running_through(IMAGE_IDX, visualise=False)
result3 = out.test_delay_and_sum_reconstruction_bandpass_envelope_is_running_through(IMAGE_IDX, visualise=False)
result4 = out.test_delay_and_sum_reconstruction_is_running_through_SCF(IMAGE_IDX, visualise=False)


plt.figure()
plt.subplot(2, 2, 1)
plt.title("Vanilla DAS")
plt.imshow(result1[:, 0, :, 0, 0].T)
plt.subplot(2, 2, 2)
plt.title("BP DAS")
plt.imshow(result2[:, 0, :, 0, 0].T)
plt.subplot(2, 2, 3)
plt.title("BP + Envelope DAS")
plt.imshow(result3[:, 0, :, 0, 0].T)
plt.subplot(2, 2, 4)
plt.title("SCF-DAS")
plt.imshow(result4[:, 0, :, 0, 0].T)
plt.tight_layout()
plt.show()

# Only uncomment the following two lines if a new reference reconstruction should be saved!
# import numpy as np
# np.savez("NUMBER_reference.npz",
#          reconstruction=result1[:, 0, :, 0, 0])
