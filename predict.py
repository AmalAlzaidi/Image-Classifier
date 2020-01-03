import argparse
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import json
import functions
import utility_functions

parser = argparse.ArgumentParser()

parser.add_argument('path_to_image', action = 'store', help= 'Enter image path:', metavar = 'path'  )
parser.add_argument('arch', action = 'store', type = str ,
                    help= 'Choose architecture:', choices=['vgg11', 'vgg13', 'vgg16'])
parser.add_argument('save_directory', action = 'store', help= 'Set directory to save checkpoints:', metavar = 'path' )
parser.add_argument('top_k', action = 'store', type = int , help= 'Choose number of top  most likely classes')
parser.add_argument('category_names', action = 'store',  default = 'cat_to_name.json', help ='Mapping of categories to real names')
parser.add_argument('gpu', action = 'store', help= 'Use GPU', choices=['GPU', 'CPU'])


args = parser.parse_args()


image_path = args.path_to_image
model_arch = args.arch
save_directory = args.save_directory
topk = args.top_k
category_names = args.category_names
gpu = args.gpu

if gpu == 'GPU':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
model = functions.load_checkpoint(model_arch,save_directory, gpu)

probs, top_classes = functions.predict(image_path, model, topk)
names = [cat_to_name[cl] for cl in top_classes]
print(probs)
print(top_classes)
print(names)