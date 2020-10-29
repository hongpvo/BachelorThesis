"""
This file store self-defined function for later use

"""

import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle
import glob
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def preprocess_physionet():
    """
    This function read and save the ECG *.mat files into pickle
    change the directory to yours when re-use
    """
    #read label
    label_df = pd.read_csv('../data/challenge2017/training2017/REFERENCE-v3.csv', header=None)
    label = label_df.iloc[:,1].values
    print(Counter(label))

    #read data
    all_data = []
    filenames = pd.read_csv('../data/challenge2017/training2017/RECORDS',header=None)
    filenames = filenames.iloc[:,0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('../data/challenge2019/training2017/{0}.mat'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data':all_data, 'label':label}
    with open('../data/challenge2017/challenge2017.pkl', 'wb') as fout:
        pickle.dump(res, fout)

def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    """
    Cut the data depending on its label as suggest by 2017 ENCASE Shenda Hong.
    Copyright 2017 Shenda Hong. All rights reserved.
    """
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def read_data_physionet(window_size=3000, stride=500):
    '''
    Load data and labels from pickle file.
    Cut the data into 30-s epoch
    Split the dataset 90-10
    Oversampled the training set
    '''

    # read pkl
    with open('../data/challenge2017/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    
    ##scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std

    ##encode label

    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    #split train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)
    print(X_train.shape, Y_train.shape)

    #slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))

    #shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]
    print(X_train.shape, Y_train.shape)
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    print(X_train.shape, Y_train.shape)
    return X_train, X_test, Y_train, Y_test, pid_test


def load_npz_file(npz_file):
    '''
    Load data and labels from a npz file.
    '''
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate

def load_npz_list_files(npz_list):
    '''
    Load data and labels from a npz file in list.
    '''
    data=[]
    labels=[]
    fs=None
    for npz_f in npz_list:
                #print("Loading {} ...".format(npz_f))
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv1d
                #tmp_data = np.squeeze(tmp_data)
                #tmp_data = tmp_data[:, :, np.newaxis]
                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)              
                tmp_data = tmp_data.squeeze()

                data.append(tmp_data)
                labels.append(tmp_labels)
    data = np.vstack(data)
    labels = np.hstack(labels)

 
  return data, labels

def load_data_eeg(ch = "fpz_cz"):
    '''
    Load data and labels from a npz file in list.
    Split the dataset 90-10
    Oversampled the training set
    '''
    npz_files = glob.glob(f"/content/drive/My Drive/Thesis/BEngThesis/data/data2017/sleep-cassette/eeg_{ch}/*.npz")
    x, y = load_npz_list_files(npz_files)
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    class_labels = np.unique(Y_train)
    n_max_classes = -1
    for c in class_labels:
            n_samples = len(np.where(Y_train == c)[0])
            if n_max_classes < n_samples:
                n_max_classes = n_samples
    print("before balance: ", Counter(Y_train))
    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(Y_train == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(X_train[idx], n_repeats, axis=0)
        tmp_y = np.repeat(Y_train[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, X_train[sub_idx]])
            tmp_y = np.hstack([tmp_y, Y_train[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    print("after balance: ", Counter(balance_y))
    print(balance_x.shape, balance_y.shape, X_test.shape, Y_test.shape)
            
    print(Counter(balance_y), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(balance_y.shape[0])
    balance_x = balance_x[shuffle_pid]
    balance_y = balance_y[shuffle_pid]

    balance_x = np.expand_dims(balance_x, 1)
    X_test = np.expand_dims(X_test, 1)
    return balance_x, X_test, balance_y, Y_test


if __name__ == "__main__":
    load_data_eeg()
