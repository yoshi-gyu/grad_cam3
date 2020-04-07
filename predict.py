import torch
import torch.nn.functional as functional
from torchvision import transforms
from torchvision import models
from grad_cam import GradCam

import urllib
import pickle

import numpy as np
from PIL import Image

import sys
sys.path.append('/home/yoshi/miniconda3/lib/python3.7/site-packages')
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def predict():

    labels = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )

    # model
    model = models.vgg19(pretrained=True)
    grad_cam = GradCam(model=model, feature_layer=list(model.features)[-1])

    # image
    VISUALIZE_SIZE = (224, 224)
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    image_transform = transforms.Compose([
        transforms.Resize(VISUALIZE_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    path = "images/guinea-pig-242520_640.jpg"
    image = Image.open(path)
    # image.thumbnail(VISUALIZE_SIZE, Image.ANTIALIAS)
    # display(image)

    image_orig_size = image.size # (W, H)
    print('image.size:', image.size)

    img_tensor = image_transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    # grad_cam
    model_output = grad_cam.forward(img_tensor)
    target = model_output.argmax(1).item()
    score = model_output.max(1)[1]
    print('target: [{}] {} ({})'.format(target, labels[target], score))

    # predict
    grad_cam.backward_on_target(model_output, target)

    # get gradient
    feature_grad = grad_cam.feature_grad.data.numpy()[0]
    weights = np.mean(feature_grad, axis=(1,2))
    feature_map = grad_cam.feature_map.data.numpy()
    grad_cam.clear_hook()

    # get CAM
    cam = np.sum((weights * feature_map.T), axis=2).T
    cam = np.maximum(cam, 0) # apply relu to cam

    # resize
    print('cam.size: ', cam.size)
    cam = cv2.resize(cam, VISUALIZE_SIZE)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(cam * 255)

    # synthesis
    activation_heatmap = np.expand_dims(cam, axis=0).transpose(1,2,0)
    org_img = np.asarray(image.resize(VISUALIZE_SIZE))
    img_with_heatmap = np.multiply(np.float32(activation_heatmap), np.float32(org_img))
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    org_img = cv2.resize(org_img, image_orig_size)

    # visualize
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(org_img)
    plt.subplot(1,2,2)
    plt.imshow(cv2.resize(np.uint8(255 * img_with_heatmap), image_orig_size))
    plt.savefig("images/result_guinea-pig-242520_640.jpg")

if __name__ == '__main__':
    predict()    