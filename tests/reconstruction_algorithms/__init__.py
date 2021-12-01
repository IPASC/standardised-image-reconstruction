"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Groehl
SPDX-License-Identifier: MIT
"""

from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import zoom


class TestClassBase():
    """
    This base class can be used for the implementation of image reconstruction test cases.
    It automatically downloads a sample IPASC-formatted HDF5 file and
    """

    def download_sample_files(self):
        download_urls = [("14o3Bi5A_OGaZd0nfcx89Vy3AijB3emLO",
                          "1oaFPFGd0wTJ35u0NCG_IgYUktJ5XgM4Y"),
                         ("1BdSLl4BSxpxXDwWcBKKVV4nHULPe7IS8",
                          "1IXq5_stsyxLjtqWYvFebzUDXxSQZiwZt"),
                         ("1jwNkiSkou8EJv7ucg3WkrIkU6Ye3Q4ut",
                          "11GoY647IodbdAEg9fMPfhboabvEVH_oh")]

        for download_url in download_urls:
            ts_path = os.path.join(self.ipasc_hdf5_file_path, f"{download_url[0]}_ipasc.hdf5")
            reco_path = os.path.join(self.ipasc_hdf5_file_path, f"{download_url[0]}_reference.npz")

            if not os.path.exists(ts_path):
                gdd.download_file_from_google_drive(file_id=download_url[0],
                                                    dest_path=ts_path,
                                                    overwrite=False)
            else:
                print(f"File {ts_path} already in folder. Did not re-download")

            if not os.path.exists(reco_path):
                gdd.download_file_from_google_drive(file_id=download_url[1],
                                                    dest_path=reco_path,
                                                    overwrite=False)
            else:
                print(f"File {reco_path} already in folder. Did not re-download")

            self.assert_file_download_successful(download_url[0], ts_path)
            self.assert_file_download_successful(download_url[1], reco_path)

    @staticmethod
    def assert_file_download_successful(download_url, file_name):
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Could not find {file_name} in the test package and could not"
                                    f"download automatically from google drive. In order to run the tests,"
                                    f"please manually download the file from "
                                    f"https://drive.google.com/file/d/{download_url}"
                                    f"/view?usp=sharing, name it {file_name},"
                                    f"and place it into the 'tests/reconstruction_algorithms' folder.")

    def __init__(self):
        self.current_hdf5_file = ""
        self.ipasc_hdf5_file_path = os.path.abspath("./")
        self.download_sample_files()

    def run_tests(self, algorithm, **kwargs):

        hdf5_files = glob.glob(os.path.join(self.current_hdf5_file, "*.hdf5"))

        for hdf5_file in hdf5_files:
            result = algorithm.reconstruct_time_series_data(hdf5_file, **kwargs)
            reference = np.load(hdf5_file.replace("_ipasc.hdf5", "_reference.npz"))["reconstruction"]
            self.visualise_result(result, reference)

    def visualise_result(self, result: np.ndarray, reference: np.ndarray):
        result = result[:, 0, :, 0, 0]
        if len(np.shape(reference)) == 3:
            reference = reference[0, :, :]
        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        plt.title("Reference Reconstruction [a.u.]")
        plt.axis("off")
        plt.imshow(reference)
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Reconstruction Result [a.u.]")
        plt.imshow(result)
        plt.colorbar()
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Relative difference [%]")
        cur_shape = np.asarray(np.shape(result))
        tar_shape = np.asarray(np.shape(reference))
        result = zoom(result, tar_shape/cur_shape)
        plt.imshow(np.abs(reference - result) / reference * 100, cmap="Reds", vmin=0, vmax=100)
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()