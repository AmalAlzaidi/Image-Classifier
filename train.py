import argparse
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

import functions 
import utility_functions

parser = argparse.ArgumentParser(description= 'Train model',
                                epilog = "The model has been trained")

parser.add_argument('data_directory', action = 'store', help= 'Enter data directory path'  )
parser.add_argument('save_directory', action = 'store', help= 'Set directory to save checkpoints' )
parser.add_argument('arch', action = 'store', type = str ,
                    help= 'Choose architecture:', choices=['vgg11', 'vgg13', 'vgg16'])
parser.add_argument('learning_rate', action = 'store',  type = float , help= 'Choose learning rate')
parser.add_argument('hidden_units', action = 'store',  type = int , help= 'Choose number of hidden units')
parser.add_argument('epochs', action = 'store',  type = int , help= 'Choose number of hidden epochs')
parser.add_argument('gpu', action = 'store', help= 'Use GPU', choices=['GPU', 'CPU'])


args = parser.parse_args()

#Save user's inputs to use them as parameters
data_directory = args.data_directory
save_directory = args.save_dir
model_arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu 

if gpu == 'GPU':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
train_data, trainloader, validloader, testloader = utility_functions.loading_data(data_directory)
model, criterion, optimizer  = functions.Network(model_arch, hidden_units, learning_rate)
functions.train_model(epochs, model, trainloader, criterion, optimizer, validloader, gpu)
functions.save_model(model, epochs, hidden_units, optimizer, train_data, save_directory)