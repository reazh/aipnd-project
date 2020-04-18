import argparse

import utility_functions
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import nnetwork
from nnetwork import init_dataloaders, get_model, get_classifier, test_model, save_checkpoint

import workspace_utils
from workspace_utils import active_session

parser = argparse.ArgumentParser(description='Train the model with Images')
parser.add_argument('datadir', type=str, help='directory with dataset')
parser.add_argument('--save_dir', nargs='?', help='save directory')
parser.add_argument('--learning_rate', nargs='?', type=float, default=0.001, 
                    help='set the learning rate')
parser.add_argument('--arch', nargs='?', type=str, choices=['vgg16', 'alexnet','densenet121'], 
                    default='vgg16', help='select a pretrained network')
parser.add_argument('--hidden_units', nargs='?', type=int, default=512,
                    help='select the number of hidden units')
parser.add_argument('--epochs', nargs='?', type=int, default=3, 
                    help='specify the number of epochs to train over')
parser.add_argument('--gpu', action='store_true', help='enable training on GPU')

args = parser.parse_args()

if args.gpu:
    # Need to test for GPU availability
    if torch.cuda.is_available():
        device='cuda'
else: 
    device='cpu'


# Setup datasets
data_dir = args.datadir
(train_loader, valid_loader, test_loader) = init_dataloaders(data_dir)

    
arch=args.arch
hidden_units=args.hidden_units
model, input_size=get_model(arch)
model.classifier = get_classifier(input_size, hidden_units)    

learnrate=args.learning_rate
epochs=args.epochs

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
        for (images, labels) in train_loader:
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
                    
# Test your network on the Test dataset
model.eval()
with torch.no_grad():
    test_loss, accuracy = test_model(model, test_loader, criterion, "Test", device)
    print( "Test Loss: {:.3f}".format(test_loss/len(test_loader)),
           "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

# Save the model 
filename = arch + '.pth'
save_checkpoint(model, filename)

