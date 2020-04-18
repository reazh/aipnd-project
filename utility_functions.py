from PIL import Image

import numpy as np
import torch 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mean = 0
    std = 1
    im = Image.open(image)
    size = 256, 256
    if im.width > im.height:
        ratio = float(im.width) / float(im.height)
        newwidth = ratio * size[0]
        im_resized = im.resize((int(np.floor(newwidth)), size[1]), Image.ANTIALIAS)
    else:
        ratio = float(im.height)/ float(im.width)
        newheight = ratio * size[1]
        im_resized = im.resize((size[0], int(np.floor(newheight))), Image.ANTIALIAS)
    ## Calculate for the other case
    im_cropped = im_resized.crop((16, 16, 240, 240))
    # Convert PIL image to Numpy Array
    im_np = np.array(im_cropped)
    # Get the value between 0-1 by dividing by 255
    im_np = im_np / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_np_norm = (im_np - mean)/std
    im_np_norm_t = im_np_norm.transpose(2, 0, 1)
    im.close()
    return torch.Tensor(im_np_norm_t)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)
    ax.imshow(image)
    
    
    return ax

