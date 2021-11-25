from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
import matplotlib.pyplot as plt

# #####################################################################
# TEST IMAGES DOCUMENTATION
# All images are distributed via the MIT license
# #####################################################################
# #####################################################################
#
# IMAGE_IDX = 0
# Simulated image of two tubular structured underneath a horizontal layer.
#
# #####################################################################
#
# IMAGE_IDX = 1
# Simulated image of point sources in a homogeneous medium
#
# #####################################################################
#
IMAGE_IDX = 2
# Experimental image provided bei Manojit Pramanik. It is a point absorber
# in a homogeneous medium.
#
# #####################################################################

out = TestDelayAndSum("TestDelayAndSum")
out.p_factor = 1.5
out.fnumber = 0.5
result1 = out.test_delay_and_sum_reconstruction_is_running_through(IMAGE_IDX, visualise=False)
result2 = out.test_delay_and_sum_reconstruction_is_running_through_fnumber(IMAGE_IDX, visualise=False)
result3 = out.test_delay_and_sum_reconstruction_is_running_through_pDAS(IMAGE_IDX, visualise=False)
result4 = out.test_delay_and_sum_reconstruction_is_running_through_SCF(IMAGE_IDX, visualise=False)

# We can use the pDAS version for both the fnumber only or pDAS version
#out.p_factor = 1
#out.test_delay_and_sum_reconstruction_is_running_through_pDAS()

plt.figure()
plt.subplot(2, 2, 1)
plt.title("Vanilla DAS")
plt.imshow(result1[:, 0, :, 0, 0])
plt.subplot(2, 2, 2)
plt.title("fnumber DAS")
plt.imshow(result2[:, 0, :, 0, 0])
plt.subplot(2, 2, 3)
plt.title("pDAS")
plt.imshow(result3[:, 0, :, 0, 0])
plt.subplot(2, 2, 4)
plt.title("SCF-DAS")
plt.imshow(result4[:, 0, :, 0, 0])
plt.show()
