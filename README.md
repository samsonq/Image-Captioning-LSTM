# CSE 151B PA4!

This PA was completed by Siddharth Saha, Shubham Kaushal, Samson Qian and Alejandro Fosado. In this PA we implement an encoder and decoder setup using 2 primary architectures. The architectures are detailed in our report. To run an experiment with an architecture you need to follow the below format and put it in a .json file

## Format

{
#Name of the experiment. This will generate a folder with the same name under experiment_data
  "experiment_name": "baseline",
  #Links to the datasets. You can update this to reflect your paths and choices depending on computer specifications
  "dataset": {
    "training_ids_file_path": "./train_ids.csv",
    "validation_ids_file_path": "./val_ids.csv",
    "test_ids_file_path": "./test_ids.csv",
    "training_annotation_file_path": "./data/annotations/captions_train2014.json",
    "test_annotation_file_path": "./data/annotations/captions_val2014.json",
    "images_root_dir": "./data/images/",
    "vocabulary_threshold": 2,
    "img_size": 256,
    "batch_size": 64,
    "num_workers": 8
  },
  #Some of our core hyperparameters. Adjust to your choosing
  "experiment": {
    "num_epochs": 10,
    "learning_rate": 5e-4,
    "dropout": 0
  },
  #Same as above. Importantly model type has 3 choices: LSTM, LSTM2 and RNN. Anything else will give an error
  "model": {
    "hidden_size": 512,
    "embedding_size": 300,
    "layers": 1,
    "model_type": "LSTM"
  },
  #Controls how the captions get generated. Temperature won't matter if you have deterministic as true
  "generation": {
    "max_length": 20,
    "deterministic": true,
    "temperature": 0
  }
}

## How to run
Say you saved a json file as new_file.json

Just say python main.py new_file

A folder will generated under experiment_data in which you can see your performance and the training process results. If you want to see the some visualizations while testing you can run this in a notebook instead by saying !python main.py new_file and it should show you pictures, actual captions and our predicted caption after training


## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
