################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from caption_utils import bleu1, bleu4
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
config = read_file("./default.json")

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
    extras = {"num_workers": config["dataset"]["num_workers"], "pin_memory": True}
else:
    device = torch.device("cpu")
    extras = False


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./model_configs/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(self.__vocab, config_data)

        self.__criterion = nn.CrossEntropyLoss()
        params = list(self.__model[1].parameters()) + list(self.__model[0].fc.parameters()) #+ list(self.__model[0].bn.parameters())
        self.__optimizer = torch.optim.Adam(params, lr=config_data["experiment"]["learning_rate"])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)
            state_dict_encoder = torch.load(os.path.join(self.__experiment_dir, 'latest_model_encoder.pt'))
            state_dict_decoder = torch.load(os.path.join(self.__experiment_dir, 'latest_model_decoder.pt'))
            self.__model[0].load_state_dict(state_dict_encoder['model'])
            self.__model[1].load_state_dict(state_dict_decoder['model'])
            self.__optimizer.load_state_dict(state_dict_encoder['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model[0] = self.__model[0].cuda().float()
            self.__model[1] = self.__model[1].cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model[0].train()
        self.__model[1].train()
        training_loss = 0

        for i, (images, captions, _) in enumerate(self.__train_loader):
            images = images.to(device)
            captions = captions.to(device)
            self.__model[1].zero_grad()
            self.__model[0].zero_grad()
            features = self.__model[0](images)
            outputs = self.__model[1](features, captions)
            loss = self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
            loss.backward()
            self.__optimizer.step()
            training_loss += loss.item()

        return training_loss/len(self.__train_loader)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model[0].eval()
        self.__model[1].eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
                features = self.__model[0](images)
                outputs = self.__model[1](features, captions)
                loss = self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
                val_loss += loss.item()

        return val_loss/len(self.__val_loader)
    
    def clean_caption(self, pred_caption):
        pred_caption = pred_caption[0].cpu().numpy()

        sampled_caption = []
        for word_id in pred_caption:
            word = self.__vocab.idx2word[word_id].lower()
            if word == '<start>' or word == '.':
                continue
            elif word == '<end>':
                break
            sampled_caption.append(word)
        #return ' '.join(sampled_caption)
        return sampled_caption

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self, verbose=1):
        self.__model[0].eval()
        self.__model[1].eval()
        test_loss = 0
        bleu_1 = 0
        bleu_4 = 0

        with torch.no_grad():
            for iter, (images, captions, img_ids) in tqdm(enumerate(self.__test_loader), total=len(self.__test_loader)):
                images = images.to(device)
                captions = captions.to(device)
                features = self.__model[0](images)
                outputs = self.__model[1](features, captions)
                loss = self.__criterion(outputs.view(-1, len(self.__vocab)), captions.view(-1))
                test_loss += loss.item()
                
                pred_caption = self.__model[1].sample(features)
                pred_caption = self.clean_caption(pred_caption)
                captions = self.clean_caption(captions)
                if verbose:
                    print('predicted caption:', ' '.join(pred_caption))
                    print('actual caption:', ' '.join(captions))
                for _ in range(len(captions) - len(pred_caption)):
                    pred_caption.append("")
                bleu_1 += bleu1(captions, pred_caption)
                bleu_4 += bleu4(captions, pred_caption)

        test_loss /= len(self.__test_loader)
        bleu_1 /= len(self.__test_loader)
        bleu_4 /= len(self.__test_loader)
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                               bleu_1,
                                                                               bleu_4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path_encoder = os.path.join(self.__experiment_dir, 'latest_model_encoder.pt')
        root_model_path_decoder = os.path.join(self.__experiment_dir, 'latest_model_decoder.pt')
        model_dict_encoder = self.__model[0].state_dict()
        model_dict_decoder = self.__model[1].state_dict()
        state_dict_encoder = {'model': model_dict_encoder, 'optimizer': self.__optimizer.state_dict()}
        state_dict_decoder = {'model': model_dict_decoder, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict_encoder, root_model_path_encoder)
        torch.save(state_dict_decoder, root_model_path_decoder)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
