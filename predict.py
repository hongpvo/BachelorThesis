"""
This file use to load the model at specific epoch and evaluate it
Draw the normalized confusion matrix

To use this file please run
!python predict.py
"""
import itertools
import sys
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import read_data_physionet, load_data_eeg
from deep_nn import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
#from torchsummaryX import summary
from torchsummary import summary

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Define the confusion matrix plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    task = "ECG" #choose the task, EEG or ECG
    LSTM = True
    #load the test set
    if task == "ECG":
        X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet()
    elif task == "EEG":
        X_train, X_test, Y_train, Y_test = load_data_eeg()

    dataset_test = MyDataset(X_test, Y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

    #create the corresponding model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    model = ResNet1D(
        in_channels=1,
        base_filters=128, #64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size,
        stride=stride,
        groups=32,
        n_block=n_block,
        n_classes=4,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap,
        use_do=True,
        lstm = LSTM)
    #load the model to GPU
    model.to(device)

    #set training params
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    #Load model at specific epoch
    checkpoint = torch.load("model/model3.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    scheduler.load_state_dict(checkpoint['scheduler'])


    #evaluate the model
    model.eval()
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)

        if task == "ECG":
            ##vote most common
            final_pred = []
            final_gt = []
            for i_pid in np.unique(pid_test):
                tmp_pred = all_pred[pid_test==i_pid]
                tmp_gt = Y_test[pid_test==i_pid]
                final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
                final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
        #plot the confusion matrix
        if task == "ECG":
            cfm = confusion_matrix(final_gt, final_pred)
            print(cfm)
            plt.figure(figsize=(4,4))
        elif task == "EEG":
            cfm = confusion_matrix(Y_test, all_pred)
            print(cfm)
            plt.figure(figsize=(5,5))
        plot_confusion_matrix(cfm, class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()