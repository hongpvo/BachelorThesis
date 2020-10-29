"""
This function use to train and validate the model
The log and result will be saved by summarywriter
The model with its epoch, loss, optimizer and scheduler will be saved for later resuming and testing

To train please run:
!python train.py

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
from torchsummary import summary #if use LSTM please use torchsummaryX instead

if __name__ == "__main__":
    is_debug = False
    batch_size = 128        #set batch size
    task = "ECG"            #set task, ECG or EEG
    LSTM = True             #turn LSTM on or off
    n_classes = 4

    #create log file
    if is_debug:
        writer = SummaryWriter("debug")
    else:
        writer = SummaryWriter("train_resNet/run_128lstm_4_0_12")

    #load dataset and set the number of classes, 5 for EEG and 4 for ECG
    if task == "ECG":
        X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet()
        n_classes = 4
    elif task == "EEG":
        X_train, X_test, Y_train, Y_test = load_data_eeg()
        n_classes = 5

    print(X_train.shape, Y_train.shape)
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

    #create model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    kernel_size = 16
    stride = 2
    n_block = 48 #15 for 33-model
    downsample_gap = 6 #2 for 33-model
    increasefilter_gap = 12 #4 for 33-model
    model = ResNet1D(
        in_channels=1,
        base_filters=128, #set the number of base filters, 64*i
        kernel_size=kernel_size,
        stride=stride,
        groups=32,
        n_block=n_block,
        n_classes=n_classes,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap,
        use_do=True,
        lstm = LSTM)

    if LSTM:
        summary(model, torch.zeros(1,1,3000))
    else:
        summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    #load the model to GPU
    model.to(device)

    #set up loss function and optimizer, learning rate as well as gradient clipping
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3) #change lr=1e-4 if the performance is low
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    epoch = 0

    #Load model if resuming training
    checkpoint = torch.load("/content/drive/My Drive/Thesis/BEngThesis/ECG/model/model_128lstm_4/model_128lstm_4-epoch11.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint(epoch {})".format(checkpoint['epoch']))


    n_epoch = 50 #set the number of epoch
    step = 0

    for _ in tqdm(range(epoch+1,epoch+1+n_epoch), desc='epoch', leave=False):
        #train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break

        scheduler.step(_)

        #test
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
            ##classification report
            tmp_report = classification_report(final_gt, final_pred, output_dict=True)
            print(confusion_matrix(final_gt, final_pred))
            f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
            writer.add_scalar('F1/f1_score', f1_score, _)
            writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
            writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
            writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], _)
            writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], _)

        #save the model for later use
        torch.save({
            'epoch': _,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'loss': loss
            }, "/content/drive/My Drive/Thesis/BEngThesis/ECG/model/model_128lstm_4/model_128lstm_4-epoch{}.pth".format(_))