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



GroupPhotoName="InputImage\\GroupPhotoDownscaled.png"



# deleting the files from DetectedFaces
pattern = "DetectedFaces\\*"
files = glob.glob(pattern)
for file in files:
    os.remove(file)

# deleting the files from UpscaledFaces
pattern = "UpscaledFaces\\*"
files = glob.glob(pattern)
for file in files:
    os.remove(file)
    
# load SSD and ResNet network based caffe model for 300x300 dim imgs
net = cv2.dnn.readNetFromCaffe("Caffemodel\\weights-prototxt.txt", "Caffemodel\\res_ssd_300Dim.caffeModel")


# load the input image by resizing to 300x300 dims
image = cv2.imread(GroupPhotoName)
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# pass the blob into the network
net.setInput(blob)
detections = net.forward()

count=1
# loop over the detections to extract specific confidence
for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # greater than the minimum confidence
    if confidence > 0.2:
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x1, y1, x2, y2) = box.astype("int")
 
        y1=y1-30 if y1-30 >0 else 1
        x1=x1-30 if x1-30>0 else 1
        x2=x2+30
        y2=y2+30
        crop=image[y1:y2,x1:x2]
        name="DetectedFaces\\face"+str(count)+".png"
        count=count+1
        cv2.imwrite(name,crop)

UPSCALE_FACTOR = 4
image_filenames = [join("DetectedFaces", x) for x in listdir("DetectedFaces")]

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
    out_img.save("UpscaledFaces\\"+imageName.split("\\")[1])


# In[ ]:




