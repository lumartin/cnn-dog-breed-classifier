from imgaug import augmenters as iaa
import imgaug as ia
import PIL
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class ImgAugTransform:
  
  def __init__(self, augmentationList):
    self.aug = iaa.Sequential(augmentationList)

  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)


def show_sample(dataset, rows=6, columns=1):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(rows)))
                   for i in range(columns)))
  plt.imshow(img)
  plt.axis('off')

def default_transformation():
  return transforms.Compose([transforms.Resize(size=224),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

def mixed_transformation(customTransformations):
  return torchvision.transforms.Compose([
                    customTransformations,
                    lambda x: PIL.Image.fromarray(x),
                    torchvision.transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
                    ])
