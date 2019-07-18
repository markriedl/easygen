import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import re

######################################
## GLOBALS

# desired depth layers to compute style/content losses :
CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])



#######################################
## CLASSES

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

#######################################
## HELPERS

def tensor_to_image(tensor):
    t = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = t(image)
    return image

def image_loader(image_name, size, device):
    # transform images to same size
    t = transforms.Compose([transforms.Resize(size),  # scale imported image
                            transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = image.resize((size, size), PIL.Image.ANTIALIAS)
    image = t(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, device,
                               content_layers = CONTENT_LAYERS_DEFAULT,
                               style_layers = CONTENT_LAYERS_DEFAULT):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

###########################################

def run_style_transfer(cnn, content_img, style_img, input_img, device, 
                       normalization_mean = CNN_NORMALIZATION_MEAN, 
                       normalization_std = CNN_NORMALIZATION_STD,
                       content_layers = CONTENT_LAYERS_DEFAULT, 
                       style_layers = STYLE_LAYERS_DEFAULT,
                       num_steps = 300,
                       style_weight = 1000000, 
                       content_weight = 1
                       ):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean.to(device), normalization_std.to(device), style_img, content_img, device,
        content_layers, style_layers)
    optimizer = get_input_optimizer(input_img)
    
    best_img = [None]
    best_score = [None]

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            current_score = style_score.item() + content_score.item()
            if best_img[0] is None or current_score <= best_score[0]:
                best_img[0] = input_img.clone()
                best_img[0].data.clamp_(0, 1) 
                best_score[0] = current_score

            return style_score + content_score

        optimizer.step(closure)
    # a last correction... # not sure I need to do this
    input_img.data.clamp_(0, 1)
    
    return best_img[0]

##########################################
def process_layers_spec(spec):
    spec = str(spec)
    layers = re.findall(r'[\-0-9]+', spec)
    layers = [int(num) for num in layers]
    if -1 in layers or 0 in layers:
        return STYLE_LAYERS_DEFAULT
    layers = list(filter(lambda x: x >= 1 and x <= 5, layers))
    layers = sorted(layers)
    layers = ['conv_' + str(num) for num in layers]
    return layers

##########################################

def run(content_image_path, style_image_path, output_path,
        image_size = 512, num_steps = 300, style_weight = 1000000, content_weight = 1, 
        content_layers_spec='4', 
        style_layers_spec = '1, 2, 3, 4, 5'):
    # CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # process content layers specification
    content_layers = process_layers_spec(content_layers_spec)
    style_layers = process_layers_spec(style_layers_spec)
    
    # Load images
    style_img = image_loader(style_image_path, image_size, device)
    content_img = image_loader(content_image_path, image_size, device)
 
    assert style_img.size() == content_img.size(), "style and content images must be the same size"

    # Load the VGG CNN. Download if necessary.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    output = run_style_transfer(cnn, 
                                content_img, style_img, input_img, device,
                                num_steps = num_steps,
                                style_weight = style_weight,
                                content_weight = content_weight,
                                content_layers = content_layers,
                                style_layers = style_layers)
    img = tensor_to_image(output)
    img.save(output_path, "JPEG")
    return img

