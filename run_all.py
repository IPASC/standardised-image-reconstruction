"""
import image_reconstruction.baseline_delay_and_sum as das

das.
out = TestDelayAndSum()
out.test_delay_and_sum_reconstruction_is_running_through()


from tests.reconstruction_algorithms import TestClassBase
import unittest

#//a = unittest.TestCase("test")
t = TestClassBase()
"""
import matplotlib.pyplot as plt
from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
out1 = TestDelayAndSum().test_delay_and_sum_reconstruction_is_running_through()
out2 = TestDelayAndSum().test_delay_and_sum_reconstruction_is_running_through_fnumber()
plt.show()
