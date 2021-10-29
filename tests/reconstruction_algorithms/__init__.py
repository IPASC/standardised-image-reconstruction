"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-License-Identifier: MIT
"""

import unittest
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio


class TestClassBase(unittest.TestCase):
    """
    This base class can be used for the implementation of image reconstruction test cases.
    It automatically downloads a sample IPASC-formatted HDF5 file and
    """

    def download_sample_ipasc_file(self):
        download_url = "1RzzNrwGewkyX8beVoi7zHiQ4BxdUkbRQ"

        if not os.path.exists(self.file_name):
            gdd.download_file_from_google_drive(file_id=download_url, dest_path=self.ipasc_hdf5_file_path)
        else:
            print(f"File {self.file_name} already in folder. Did not re-download")

        if not os.path.exists(self.file_name):
            raise FileNotFoundError(f"Could not find {self.file_name} in the test package and could not"
                                    f"download automatically from google drive. In order to run the tests,"
                                    f"please manually download the file from "
                                    f"https://drive.google.com/file/d/{download_url}"
                                    f"/view?usp=sharing, name it {self.file_name},"
                                    f"and place it into the 'tests/reconstruction_algorithms' folder.")

    def download_expected_result_file(self):
        download_url = "1l_9QTqCcIqW58d_YHPcTixUyB0OrSkq5"

        if not os.path.exists(self.file_name_result):
            gdd.download_file_from_google_drive(file_id=download_url, dest_path=self.result_file_path)
        else:
            print(f"File {self.file_name_result} already in folder. Did not re-download")

        if not os.path.exists(self.file_name_result):
            raise FileNotFoundError(f"Could not find {self.file_name_result} in the test package and could not"
                                    f"download automatically from google drive. In order to run the tests,"
                                    f"please manually download the file from "
                                    f"https://drive.google.com/file/d/{download_url}"
                                    f"/view?usp=sharing, name it {self.file_name_result},"
                                    f"and place it into the 'tests/reconstruction_algorithms' folder.")

    def __init__(self, arg):
#        super(TestClassBase, self).__init__(arg)
        self.file_name = "sample_file.hdf5"
        self.file_name_result = "sample_result.png"
        self.ipasc_hdf5_file_path = os.path.abspath("./"+self.file_name)
        self.result_file_path = os.path.abspath("./" + self.file_name_result)
        self.download_sample_ipasc_file()
        self.download_expected_result_file()

    def setUp(self) -> None:
        self.download_sample_ipasc_file()

    def visualise_result(self, result: np.ndarray):
        expected_result = imageio.imread(self.result_file_path)
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.title("Reference Reconstruction")
        plt.axis("off")
        plt.imshow(expected_result)
        plt.subplot(1, 2, 2)
        plt.title("Reconstruction Result")
        plt.imshow(np.rot90(result[:, 0, :, 0, 0], -1))
        plt.axis("off")
        plt.tight_layout()
        #plt.show()
        #plt.close()
