import os
import numpy as np
import torch
import torch.nn as nn


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
