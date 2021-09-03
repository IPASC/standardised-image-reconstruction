"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-License-Identifier: MIT
"""

import unittest
from google_drive_downloader import GoogleDriveDownloader as gdd
import os


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
                                    f"please manually download the file from {download_url}, name it {self.file_name},"
                                    f"and place it into the 'tests/reconstruction_algorithms' folder.")

    def __init__(self, arg):
        super(TestClassBase, self).__init__(arg)
        self.file_name = "sample_file.hdf5"
        self.ipasc_hdf5_file_path = os.path.abspath("./"+self.file_name)
        self.download_sample_ipasc_file()

    def setUp(self) -> None:
        self.download_sample_ipasc_file()