# IPASC Standardised Image Reconstruction Project

Code repository for the IPASC standardised image reconstruction project

The project is divided into three tasks:

- Data generation and collection
- Implementation of reconstruction algorithms
- Development of an evaluation framework

Each of these tasks has one module in this repository where the code is collected.
How these modules are intended to be used is described in the following sections:

## Getting Started

0. Install [git](https://git-scm.com/downloads) and [python3](https://www.python.org/downloads/) for your platform.
1. Clone the repository with `git clone https://github.com/IPASC/standardised-image-reconstruction.git`.
2. Enter the directory: `cd standardised-image-reconstruction` (Linux and Mac).
3. **Optional** [Create a virtual enviroment](https://docs.python.org/3/library/venv.html) with `python3 -m venv env` and enter it with `source env/bin/activate` (Linux and Mac).
4. [Install dependences](https://pip.pypa.io/en/stable/cli/pip_install/?highlight=requirements) `pip install -r requirements.txt`
5. Make sure everything works by running `python3 run_all.py`

## Data Generation

Data simulaion will be done using optical and acoustic forward modelling.

### Optical Modelling

TODO

### Acoustic Modelling

For the  acoustic forward model, we are using k-Wave (http://k-wave.org/). 
A base script for the simulation can be found at `data_generation/base_script/`.
The contained scripts are meant to be used as a guide to facilitate setting up a k-Wave
simulation on a new data set.

The script also enables a data export into the IPASC data format.
To enable this feature, the latest version of the PACFISH tool needs to be downloaded and
the path to the `PATH/TO/PACFISH/pacfish_matlab` folder has to be added as an absolute path
to the MATLAB paths (please adjust the path based on where the folder is on your computer).

Please download the pacfish tool from github (https://github.com/IPASC/PACFISH/):
  
  `git clone https://github.com/IPASC/PACFISH/`

The resulting data should in the end be uploaded to the IPASC Google Drive using this 
link: https://drive.google.com/drive/folders/19ZnxzWITQl7K9sCQsF1TxYfChYMYSs89

### Experimental Data

TODO

## Image Reconstruction

Each algorithm implementation should be done in a separate class that inherits from 
the `ReconstructionAlgorithm` base class defined in `image_reconstruction.__init__.py`.
The `BaselineDelayAndSumAlgorithm` can be used as a reference.

There exists a test base class that automatically downloads sample_data in the IPASC format such
that the algorithm implementation can easily be tested. Please refer to the `test_baseline_delay_and_sum.py`
file for a coding reference.

## Quality Assessment

**TODO**

## Contribute to the project

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for information on how to contribute.
