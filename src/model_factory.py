################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from file_utils import read_file
from tqdm import tqdm

config = read_file("./model_configs/default.json")

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
    extras = {"num_workers": config["dataset"]["num_workers"], "pin_memory": True}
else:
    device = torch.device("cpu")
    extras = False


class Encoder(nn.Module):
    def __init__(self, embed_size=config["model"]["embedding_size"],):
        super(Encoder, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  Batch Normalization
        self.initialize_weights()

    def initialize_weights(self):
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        #features = torch.autograd.Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=config["model"]["embedding_size"],
                 num_layers=config["model"]["layers"],
                 hidden_size=config["model"]["hidden_size"],
                 model_type=config["model"]["model_type"],
                 dropout=config["experiment"]["dropout"]):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if model_type == "LSTM":
            self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif model_type == "RNN":
            self.decoder = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.initialize_weights()

    def initialize_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeds = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        out, hidden = self.decoder(inputs)
        return self.fc(out)

    def deterministic_sample(self, features, states=None, sample_length=config["generation"]["max_length"]):
        sampled = None
        states = None
        inputs = features.unsqueeze(1)
        #+2 to account for start and end tokens
        for _ in range(sample_length+2):
            if states is None:
                hidden, states = self.decoder(inputs)
            else:
                hidden, states = self.decoder(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.fc(hidden.squeeze(1))  # (batch_size, vocab_size)
            predicted = torch.argmax(outputs, dim=1).view(-1, 1)
            if sampled is None:
                sampled = predicted
            else:
                sampled = torch.cat((sampled, predicted), dim=1)
            inputs = self.embed(predicted)
        return sampled


# Build and return the model here based on the configuration.
def get_model(vocab, config_data=config):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab), embed_size=embedding_size,
                      hidden_size=hidden_size, model_type=model_type).to(device)
    return [encoder, decoder]
