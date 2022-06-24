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
3. **Optional** [Create a virtual environment](https://docs.python.org/3/library/venv.html) with `python3 -m venv env` and enter it with `source env/bin/activate` (Linux and Mac).
4. [Install dependencies](https://pip.pypa.io/en/stable/cli/pip_install/?highlight=requirements) `pip install -r requirements.txt`
5. Make sure everything works by running `python3 run_all.py`

## Data Generation


### Simulated Data

This part describes the steps that need to be done in order to simulate the virtual phantoms
used for this project.

#### Prerequisites

Do run the data modelling, you will have to manually download both the [SIMPA](https://github.com/IMSY-DKFZ/simpa) and 
the [PACFISH](https://github.com/IPASC/PACFISH) GitHub repositories. These folders have to be located within the
same parent folder on your hard drive, because of the way the code is currently set up. Please see and follow the SIMPA
and PACFISH install instructions on their respective GitHub pages.

#### Optical Modelling

Optical modelling is optional. If you want to use optical modelling, you can use the pre-implemented
simulation module in SIMPA that uses [Monte Carlo eXtreme](https://github.com/fangq/mcx), maintained by Qianqian Fang.

MCX requires the availability of an NVIDIA GPU on your computer. You can download pre-compiled 
binaries fitting your system, or compile mcx yourself from the source code [here](https://sourceforge.net/projects/mcx/files/mcx%20source/).

#### Acoustic Modelling

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

## Citations

### FFT-Based Jaeger
- Jaeger, M., Schüpbach, S., Gertsch, A., Kitz, M., & Frenz, M. (2007). Fourier reconstruction 
  in optoacoustic imaging using truncated regularized inverse k-space interpolation. Inverse Problems, 
  23(6), S51.
### FFT-Based Hauptmann 
- Hauptmann, A., Cox, B., Lucka, F., Huynh, N., Betcke, M., Beard, P., & Arridge, S. 
  (2018, September). Approximate k-space models and deep learning for fast photoacoustic 
  reconstruction. In International Workshop on Machine Learning for Medical Image Reconstruction 
  (pp. 103-111). Springer, Cham.
  
- Köstli, K. P., Frenz, M., Bebie, H., & Weber, H. P. (2001). Temporal backward projection 
  of optoacoustic pressure transients using Fourier transform methods. Physics in Medicine 
  & Biology, 46(7), 1863.

- Xu, Y., Feng, D., & Wang, L. V. (2002). Exact frequency-domain reconstruction for 
  thermoacoustic tomography. I. Planar geometry. IEEE transactions on medical imaging, 
  21(7), 823-828.
  
### Back Projection
- Xu, M., & Wang, L. V. (2005). Universal back-projection algorithm for photoacoustic computed 
  tomography. Physical Review E, 71(1), 016706.
  
### Delay-Multiply-and-Sum
- Matrone, Giulia, Alessandro Stuart Savoia, Giosuè Caliano, and Giovanni Magenes. 
  "The delay multiply and sum beamforming algorithm in ultrasound B-mode medical imaging."
  IEEE transactions on medical imaging 34, no. 4 (2014): 940-949.
