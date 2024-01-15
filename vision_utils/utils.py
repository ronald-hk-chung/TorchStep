import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_batch(dataloader: torch.utils.data.DataLoader,
               transforms: transforms.Compose = T.ToPILImage(),
               labelling: Callable):
  """Function to show a batch of images with custom labelling
  
  Args:
    dataloader [DataLoader]: DataLoader of images to show
    transforms [transforms]: transform of image tensor to PIL
    labelling [Callable]: function to return label given labels generated from dataloader
  """
  fig = plt.figure(figsize=(20, 10))
  imgs, *labels = next(iter(dataloader))
  nrows, ncolumns = 4, 8
  for i, label in enumerate(list(zip(*labels))):
    plt.subplot(nrows, ncolumns, i + 1)
    plt.imshow(transforms(imgs[i]))
    plt.title(labelling(label))
    plt.axis(False)


def show(img, bbs=None):
  """Plot image 
  
  Args:
    img [PIL.Image]: image to show
    bbs [list]: list bounding boxes of [(x, y, w, h)]
  """
  fig, ax = plt.subplots()
  ax.imshow(img)
  if bbs is not None:
    for bb in bbs:
      x, y, w, h = bb
      rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
  plt.axis(False)
  plt.show()