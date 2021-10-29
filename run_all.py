from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
import matplotlib.pyplot as plt

out = TestDelayAndSum("DAS")
out.test_delay_and_sum_reconstruction_is_running_through()
out.test_delay_and_sum_reconstruction_is_running_through_fnumber()
plt.show()
