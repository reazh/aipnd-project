import argparse

import json

import nnetwork
from nnetwork import load_checkpoint

import utility_functions
from utility_functions import imshow, process_image

import utility_functions
import torch

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    probs = []
    classes = []
    # First process the image file into a Tensor
    img = process_image(image_path)
    img = img.to(device)
    model = model.to(device)
        
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(img)
    
    ps = torch.exp(output).to('cpu')
    classes = ps.topk(k=topk, dim=1)[1].numpy()
    probs = ps[0][classes[0]].numpy()
    
    # Classes are indexed from 0 while categories are indexed from 1
    # increment to adjust this discrepancy
    classes += 1
    return probs, classes

parser = argparse.ArgumentParser(description='Predict the category of the image')
parser.add_argument('image_path', type=str, help='Path to image')
parser.add_argument('checkpoint', type=str, help='checkpoint file')
parser.add_argument('--top_k', type=int, default=3, 
                    help='print out top k predictions')
parser.add_argument('--category_names', nargs=1, type=str, default='cat_to_name.json', 
                    help='Specify a category name map file')
parser.add_argument('--gpu', action='store_true', help='Enable inference using GPU')
     
args = parser.parse_args()                    

if args.gpu:
    # Need to test for GPU availability
    if torch.cuda.is_available():
        device='cuda'
else: 
    device='cpu'

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

model = load_checkpoint(args.checkpoint)

probs, classes = predict(args.image_path, model, args.top_k)

names = [cat_to_name[str(i)] for i in classes[0]]

print("{0:<20} {1:>15}".format("Name", "Probabilities"))
print("----------------------------------------")
for name, prob in zip(names, probs):
    print("{0:<20} {1:>15.5f}".format(name, prob))
