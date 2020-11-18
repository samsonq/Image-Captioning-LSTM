import os
import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    """

    """
    def __init__(self, classes=20):
        super(CNN, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        """
        Performs forward propagation on input.
        :param x: input
        :return: output of network
        """
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))

        return out4


class LSTM(nn.Module):
    """

    """
    def __init__(self, input_dim, neurons, layers, dropout=0.1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, neurons, layers, dropout=dropout)
        self.fc = nn.Linear(neurons, input_dim)

    def forward(self, x, hidden):
        """

        :param x:
        :param hidden:
        :return:
        """
        is_size_one = (x.shape[0] == 1)

        x, hidden = self.lstm(x, hidden)
        x = self.fc(x.squeeze())

        if is_size_one:
            return x[None, :], hidden

        return x, hidden


class VanillaRNN(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_dim, layers, dropout=0.1):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, x, hidden):
        """

        :param x:
        :param hidden:
        :return:
        """
        is_size_one = (x.shape[0] == 1)

        x, hidden = self.rnn(x, hidden)
        x = self.fc(x.squeeze())

        if is_size_one:
            return x[None, :], hidden

        return x, hidden
