
# List of contributors

| Name | Institution | Contributions |
| ---- | --- | ------------- |
| Ben Cox | University College London | Data simulation: MATLAB scripts and base phantoms |
| Kris K Dreher | German Cancer Research Center | Image Reconstruction: DMAS and sDMAS beamforming |
| Janek Gröhl | University of Cambridge | General Maintenance; Code Infrastructure; Image Reconstruction: baseline back-projection algorithm; Data simulation: SIMPA integration |
| Lina Hacker | University of Cambridge | Image Quality Measures |
| Jenni Poimala | University of Oulu | Image Reconstruction: FFT-based image reconstruction |
| Mengjie Shi | Kings College London | Image Reconstruction: FFT-based image reconstruction |
| François Varray | Creatis, Université de Lyon | Image Reconstruction: back-projection variant implementations: fnumber, pDAS, SCF, PCF; general testing |


# How to contribute

We welcome any forms of contributions to the project!
If you are unsure how to contribute, contact either Ben Cox, Lina Hacker, or
Janek Gröhl (@jgroehl) for guidance. Before contributing you should be aware of
some boundary conditions that are outlined here:

### License
The project is licensed under the MIT license.
Every contributor has to make their contributions available under the same license.
Including materials from other licenses is only possible, if the respective license is
compatible with the MIT license

### Copyright
Every contributor (or their institution) will retain their copyright.
The copyrights applicable to each file in the code will be made clear 
explicitly using the SPDX standard in a file header:

    """
    SPDX-FileCopyrightText: 2021 Random Author Name
    SPDX-License-Identifier: MIT
    """

By contributing, authors have to have the rights to actually 
contribute the code to the project and agree to the developer's
certificate of origin:

### Developer's Certificate of Origin

When contributing to the project, you agree to the following terms,
stating that you have indeed the right to contribute the code
under the MIT license and that you acknowledge that the contributed
code will be and remain publically available.

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
	    are public and that a record of the contribution (including all
	    personal information I submit with it, including my sign-off) is
	    maintained indefinitely and may be redistributed consistent with
	    this project or the open source license(s) involved.

To validate that you agree with these terms, please sign off the last commit
before your pull request, by adding the following line to the commit message:

    Signed-off-by: Random J Developer <random@developer.example.org>

This is a built-in feature of git and you can automate this by using the `-s` flag.

### Coding conventions

We ask all contributors to follow a couple of conventions and best practices when contributing code:
- Code is formatted according to the Python PEP-8 coding standard.
- Contributors create a test case that tests their code.
- Contributors document their code.

### Practical Workflow

Contributors open issue and create implementation on a separate branch or fork.
Any open questions / calls for help are addressed via a meeting taking place every second week or the comment function in the issue.
Once the contributor is happy with their code they sign-off the last commit and open a pull request.
