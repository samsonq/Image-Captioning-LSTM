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

config = read_file("./default.json")


class Encoder(nn.Module):
    def __init__(self, embed_size=config["model"]["embedding_size"],):
        super(Encoder, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def initialize_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = torch.autograd.Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size,
                 num_layers=config["model"]["layers"],
                 hidden_size=config["model"]["hidden_size"]):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def initialize_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        outputs = self.linear(hidden[0])
        return outputs

    def sample(self, features, states=None, samples=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in tqdm(range(samples)):
            hidden, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.linear(hidden.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()


# Build and return the model here based on the configuration.
def get_model(config_data, vocab, model):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")
