import torch
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

def loading_data(data_directory):
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(data_directory, transform = train_transforms)
    valid_data = datasets.ImageFolder(data_directory, transform = valid_transforms)
    test_data = datasets.ImageFolder(data_directory, transform = test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    
    return  train_data, trainloader, validloader, testloader


def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
   
    im = Image.open(image)
    im = im.resize((256, 256))
    (left, upper, right, lower) = (16, 16, 240, 240)
    im = im.crop((left, upper, right, lower))
    
    np_image = np.array(im)
    np_image = np_image/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (np_image - mean) / std
    
    image = image.transpose(2, 0, 1)
    
    return image