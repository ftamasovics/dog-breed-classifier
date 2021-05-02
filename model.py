import numpy as np
import cv2
import os
import seaborn as sns
from PIL import Image

import torch
import torchvision.models as models
from torchvision import datasets

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def ResNet50_model():
    RESNET50 = models.resnet50(pretrained=True)

    # move model to GPU if CUDA is available
    # if use_cuda:
    #     RESNET50 = RESNET50.cuda()

    RESNET50.eval()
    return RESNET50


def ResNet50_predict(img_path, RESNET50):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image

    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    RESNET50.eval()

    tensor_transforms = image_transforms(Image.open(img_path).convert('RGB'))[:3,:,:].unsqueeze(0)

    # if use_cuda:
    #     tensor_transforms = tensor_transforms.cuda()

    with torch.no_grad():
        prediction = torch.argmax(RESNET50(tensor_transforms)).item()

    RESNET50.train()
    return prediction # predicted class index

def get_model():
    ## TODO: Specify model architecture
    model_transfer = models.resnet50(pretrained=True)

    for param in model_transfer.parameters():
        param.requires_grad = False

    model_transfer.fc = nn.Linear(2048, 133)

    device = torch.device('cpu')
    # if use_cuda:
    #     model_transfer = model_transfer.cuda()

    # load the model that got the best validation accuracy (uncomment the line below)
    model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location=device))
    return model_transfer


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector_ResNet50(img_path, RESNET50):
    ## TODO: Complete the function.
    predicted_idx = ResNet50_predict(img_path, RESNET50)
    result = predicted_idx >=151 and predicted_idx <=268
    return result # true/false

def predict_breed_transfer(img_path, model_transfer, topk=5):
    data_dir = '/Users/feliciatamasovics/Documents/ML/Term2/Capstone/web_app/data/dog_images'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
                      }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                     }

    class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]


    # load the image and return the predicted breed
    image = process_image(img_path)

    #convert from numpy to tensor - torch.Size([3, 224, 224])
    np_to_tensor = torch.tensor(image).float()

    #new tensor with a dimension of size one inserted at position 0 - torch.Size([1, 3, 224, 224])
    np_to_tensor = np_to_tensor.unsqueeze(dim=0)
    model = model_transfer
    model = model.cpu()
    model.eval()

    with torch.no_grad():
        logps = model.forward(np_to_tensor)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        p = top_p.numpy()[0]
        p0=p[0]
        p1=p[1]
        p2=p[2]
        p3=p[3]
        p4=p[4]
        p01=p0/(p0+p1+p2+p3+p4)
        p11=p1/(p0+p1+p2+p3+p4)
        p21=p2/(p0+p1+p2+p3+p4)
        p31=p3/(p0+p1+p2+p3+p4)
        p41=p4/(p0+p1+p2+p3+p4)
        tot = p01+p11+p21+p31+p41
        prob = [p01, p11, p21, p31, p41]
        top_class = top_class.numpy().tolist()[0]

        top_class_names = [class_names [i] for i in top_class]

    return prob, top_class_names, top_class

def predict_breed_for_human(img, model_transfer):
    prob, top_class_names, top_class = predict_breed_transfer(img, model_transfer)
    if top_class[0] == 0 or top_class[0] <= 8:
        file=("/Users/feliciatamasovics/Documents/ML/Term2/Capstone/web_app/static/valid/00"+str(top_class[0]+1)+"."+top_class_names[0]+"/").replace(' ', '_')
    elif top_class[0] == 9 or top_class[0] <= 98:
        file=("/Users/feliciatamasovics/Documents/ML/Term2/Capstone/web_app/static/valid/0"+str(top_class[0]+1)+"."+top_class_names[0]+"/").replace(' ', '_')
    else:
        file=("/Users/feliciatamasovics/Documents/ML/Term2/Capstone/web_app/static/valid"+str(top_class[0]+1)+"."+top_class_names[0]+"/").replace(' ', '_')

    human_dog = os.listdir(file)
    print(human_dog)
    file = file+human_dog[0]
    top_class_name = top_class_names[0]

    return top_class_name, file

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Process a PIL image for use in a PyTorch model

    #py_image = image_transforms(Image.open(image).convert('RGB'))[:3,:,:]
    py_image = image_transforms(Image.open(image))
    return py_image
