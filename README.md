# IPASC Standardised Image Reconstruction Project

Code repository for the IPASC standardised image reconstruction project

The project is divided into three tasks:

- Data generation and collection
- Implementation of reconstruction algorithms
- Development of an evaluation framework

Each of these tasks has one module in this repository where the code is collected.
How these modules are intended to be used is described in the following sections:

## Data Generation

**TODO**

## Image Reconstruction

Each algorithm implementation should be done in a separate class that inherits from 
the `ReconstructionAlgorithm` base class defined in `image_reconstruction.__init__.py`.
The `BaselineDelayAndSumAlgorithm` can be used as a reference.

There exists a test base class that automatically downloads sample_data in the IPASC format such
that the algorithm implementation can easily be tested. Please refer to the `test_baseline_delay_and_sum.py`
file for a coding reference.

## Quality Assessment

**TODO**
