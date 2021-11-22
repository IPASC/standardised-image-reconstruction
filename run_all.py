from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
import matplotlib.pyplot as plt

out = TestDelayAndSum("TestDelayAndSum")
out.p_factor = 1.5
out.fnumber = 0.5
out.test_delay_and_sum_reconstruction_is_running_through()
out.test_delay_and_sum_reconstruction_is_running_through_fnumber()
out.test_delay_and_sum_reconstruction_is_running_through_pDAS()
out.test_delay_and_sum_reconstruction_is_running_through_SCF()

# We can use the pDAS version for both the fnumber only or pDAS version
#out.p_factor = 1
#out.test_delay_and_sum_reconstruction_is_running_through_pDAS()

plt.show()
