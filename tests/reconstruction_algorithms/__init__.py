"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from google_drive_downloader import GoogleDriveDownloader
import os
import matplotlib.pyplot as plt
import numpy as np


class TestClassBase:
    """
    This base class can be used for the implementation of image reconstruction test cases.
    It automatically downloads a sample IPASC-formatted HDF5 file and
    """

    def download_sample_files(self):
        self.download_urls = [("14o3Bi5A_OGaZd0nfcx89Vy3AijB3emLO",
                               "1oaFPFGd0wTJ35u0NCG_IgYUktJ5XgM4Y"),
                              ("1BdSLl4BSxpxXDwWcBKKVV4nHULPe7IS8",
                               "1IXq5_stsyxLjtqWYvFebzUDXxSQZiwZt"),
                              ("1jwNkiSkou8EJv7ucg3WkrIkU6Ye3Q4ut",
                               "11GoY647IodbdAEg9fMPfhboabvEVH_oh"),
                              ("15PPMPX__ZJQLvYSdWe5CxumvueVrizFy",
                               "19VIRW9xbqXbxmQ22Yglw_oz8gAorOaz9"),
                              ("1Om0PjyQ_8v1Ak4vIoQYGBrx1uNxAmvX-",
                               "17cruZhKispUzzqjRDmK9wQo63vItBas8"),
                              ("1Bf8Ttx5S_X44TxKCeHg5MzNZwsNRzKZU",
                               "1RtO1wPdkH1qivFXQLUyGBDGIp7VLXwID"),
                              ("10ZOB6_Y24gexHgQwMHWUMyJl6GAYppAx",
                               "1RtO1wPdkH1qivFXQLUyGBDGIp7VLXwID")
                              ]

        for download_url in self.download_urls:
            ts_path = os.path.join(self.ipasc_hdf5_file_path, f"{download_url[0]}_ipasc.hdf5")
            reco_path = os.path.join(self.ipasc_hdf5_file_path, f"{download_url[0]}_reference.npz")
            
            if not os.path.exists(ts_path):
                GoogleDriveDownloader.download_file_from_google_drive(file_id=download_url[0],
                                                                      dest_path=ts_path,
                                                                      overwrite=False)
            else:
                print(f"File {ts_path} already in folder. Did not re-download")

            if not os.path.exists(reco_path):
                GoogleDriveDownloader.download_file_from_google_drive(file_id=download_url[1],
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
        self.download_urls = []
        self.current_hdf5_file = ""
        self.ipasc_hdf5_file_path = os.path.abspath("./")
        self.download_sample_files()

    def run_tests(self, algorithm, image_idx=0, visualise=True, **kwargs):

        hdf5_file = os.path.join(self.ipasc_hdf5_file_path, self.download_urls[image_idx][0] + "_ipasc.hdf5")

        result = algorithm.reconstruct_time_series_data(hdf5_file, **kwargs)
        reference = np.load(hdf5_file.replace("_ipasc.hdf5", "_reference.npz"))["reconstruction"]
        if visualise:
            self.visualise_result(result, reference)
        return result

    @staticmethod
    def visualise_result(result: np.ndarray, reference: np.ndarray):
        result = result[:, 0, :, 0, 0]
        if len(np.shape(reference)) == 3:
            reference = reference[0, :, :]
        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.title("Reference Reconstruction [a.u.]")
        plt.imshow(reference)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Reconstruction Result [a.u.]")
        plt.imshow(result)
        plt.colorbar()

        plt.tight_layout()
        plt.show()
        plt.close()
