import torch
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import torch.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms, models
import utility_functions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Network(model, hidden_units, learning_rate):
    if model == "vgg16":
        model = models.vgg16(pretrained = True)
    elif model == "vgg13":
        model = models.vgg13(pretrained = True)
    else:
        model = models.vgg11(pretrained = True)
    
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    
    
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('Dropout', nn.Dropout(0.2)),
                                    ('fc2', nn.Linear(hidden_units, 102)),
                                    ('output',nn.LogSoftmax(dim=1))
                                    ]))

        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    model.to(device);
    return model, criterion, optimizer

    

def train_model(epochs, model, trainloader, criterion, optimizer, validloader, gpu):
    if gpu == 'GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
    epochs = 2
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #Track the loss and accuracy on the validation set   
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                    
                        valid_loss += criterion(outputs, labels).item()
                    
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()


def test_network(model, testloader, gpu):
    if gpu == 'GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            
            test_loss += criterion(outputs, labels).item()
            
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print( f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

    
def save_model(model, epochs, hidden_units, optimizer, train_data, save_directory):
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'epoch': epochs,
                  'hidden_units' : hidden_units,
                  'optimizer_state_dict': optimizer.state_dict,
                  'class_to_idx': train_data.class_to_idx,
                  }

    return torch.save(checkpoint, save_directory)

def load_checkpoint(model, save_directory, gpu):
    if gpu == 'GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    
    checkpoint = torch.load(save_directory, map_location = lambda storage, loc: storage)
    if model == "vgg16":
        model = models.vgg16(pretrained = True)
    elif model == "vgg13":
        model = models.vgg13(pretrained = True)
    elif model == "vgg11":
        model = models.vgg11(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('Dropout', nn.Dropout(0.2)),
                                    ('fc2', nn.Linear(hidden_units, 102)),
                                    ('output',nn.LogSoftmax(dim=1))
                                    ]))
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    
    model.eval()
    return model


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.cpu()
    img = utility_functions.process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze_(0)
    with torch.no_grad():
        logps = model.forward(img)
    ps = torch.exp(logps)
    
    probs, indices = torch.topk(ps, topk)
    probs = probs.tolist()[0]
    indices = indices.tolist()[0]
    
    ind_cls = []
    for item in range(len(model.class_to_idx.items())):
        ind_cls.append(list(model.class_to_idx.items())[item][0])
        
    top_classes = []   
    for i in range(topk):
        top_classes.append(ind_cls[indices[i]])
        
    
    return probs, top_classes