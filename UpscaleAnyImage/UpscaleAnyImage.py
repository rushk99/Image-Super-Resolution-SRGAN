#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import packages
import numpy as np
import cv2
import argparse
import glob
import os
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from model import Generator



GroupPhotoName="GroupPhotoDownscaled.png"


# deleting the files from UpscaledImage
pattern = "UpscaledImages\\*"
files = glob.glob(pattern)
for file in files:
    os.remove(file)
    

UPSCALE_FACTOR = 4
image_filenames = [join("InputImages", x) for x in listdir("InputImages")]

model = Generator(UPSCALE_FACTOR).eval()
model.load_state_dict(torch.load('TrainedModel/netG_epoch_4_100.pth', map_location=lambda storage, loc: storage))

for imageName in image_filenames:
    image = Image.open(imageName)
    #width, height = image.size
    #image=image.resize((width//4, height//4))
    #image.save("downscale.png")
    image = Variable(ToTensor()(image)).unsqueeze(0)
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save("UpscaledImages\\"+imageName.split("\\")[1])


# In[ ]:




