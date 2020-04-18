import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import collections
from collections import OrderedDict

def init_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_data)

    return train_loader, valid_loader, test_loader

def get_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        fc_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        fc_size = 9216
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        fc_size = 1024
    # Freeze the parameters before returning the model
    for param in model.parameters():
        param.requires_grad = False 
    return model, fc_size

def get_classifier(input_size, hidden_units):
    # Define the new classifier
    # Input size has to be passed in to match pre-trained network
    # First hidden layer is configurable
    # Remember to include Dropout probability
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drpout1', nn.Dropout(p=0.3)),
        ('fc2', nn.Linear(hidden_units, 1024)),
        ('relu2', nn.ReLU()),
        ('drpout2', nn.Dropout(p=0.3)), 
        ('logits', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier

# define a test function to use for validation and/or test
def test_model(model, dataloader, criterion, key, device):
    accuracy = 0
    test_loss = 0
    for images, labels in dataloader:
        model.to(device)
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item()
        # probabilities are the inverse log function of the output
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def train_model(model, data_loader, valid_loader,  learnrate, device, epochs):
    # Set a loss criterion and 
    # set optimizer to only optimize a classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    print_every = 20
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        steps = 0
        with active_session():
            for (images, labels) in data_loader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
       
                if (steps) % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        test_loss, accuracy = test_model(model, valid_loader, criterion, "Valid", device)
                    print("Epoch: {}/{}...".format(e+1,epochs),
                        "Training Loss: {:.3f}".format(running_loss/print_every),
                        "Validation Loss: {:.3f}".format(test_loss/len(valid_loader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                    running_loss = 0
                    model.train()
    return

def save_checkpoint(model, filename):
    # TODO: Save the checkpoint 
    checkpoint = {
        'model':model,
        'state_dict':model.state_dict(),
    }
    torch.save(checkpoint, filename)
    return

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    