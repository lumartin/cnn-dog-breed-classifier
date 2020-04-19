from imgaug import augmenters as iaa
import imgaug as ia
import PIL
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_sample(dataset, rows=6, columns=1):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(rows)))
                   for i in range(columns)))
  plt.imshow(img)
  plt.axis('off')
  
class ImgAugTransform:
  def __init__(self, augmentationList):
    self.aug = iaa.Sequential(augmentationList)

  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)
