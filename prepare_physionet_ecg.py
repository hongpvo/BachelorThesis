"""
This function use to save time-series data and its label into a pickle file for later use

*detail of preprocess_physionet could be found in util.py

The directory should be changed as your location of REFERENCE-v3.csv, RECORDS and *.mat files
in preprocess_physionet() function in util.py

"""

from util import preprocess_physionet

if __name__ == "__main__":
    preprocess_physionet()

