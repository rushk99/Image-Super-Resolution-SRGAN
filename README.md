# Image-Super-Resolution-SRGAN
  A PyTorch Implementation of Image Super Resolution using SRGAN
 based on CVPR 2017 paper :-
 [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- PyTorch
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
- opencv
```
pip install opencv-python
```
- tqdm
```
pip install tqdm
```
- Pillow
```
pip install Pillow
```

## Datasets

### Train & Validation Dataset
The train and validation datasets are sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/).
Download the dataset from [here](https://data.deepai.org/PascalVOC2012.zip) and then extract approximately 16700 images into `data/trainHR` directory and 425 images `data/valHR` directory.
Train dataset should have 16700 images and Validation dataset should have 425 images.



## Repository Tree Structure
``` 
|   .gitattributes
|   data_utils.py
|   loss.py
|   main.py
|   model.py
|   README.md
|
+---data
|   +---trainHR
|   \---valHR
|
+---epochs
|       (weightsGeneratedHere)
|
+---statistics
|       (statsGeneratedHere)
|
+---UpscaleAnyImage
|   |   model.py
|   |   UpscaleAnyImage.py
|   |   WPI.jpg
|   |
|   +---InputImages
|   |       WPI_downscaled.png
|   |
|   +---TrainedModel
|   |       netG_epoch_4_100.pth
|   |
|   \---UpscaledImages
|           WPI_downscaled.png
|
\---UpscaleFaces
    |   GroupPhoto.jpg
    |   model.py
    |   UpscaleFaces.py
    |
    +---Caffemodel
    |       res_ssd_300Dim.caffeModel
    |       weights-prototxt.txt
    |
    +---DetectedFaces
    |       (facesDetectedStoredHere)
    |
    +---InputImage
    |       GroupPhotoDownscaled.png
    |
    +---TrainedModel
    |       netG_epoch_4_100.pth
    |
    \---UpscaledFaces
            (upscaledFacesGeneratedHere)
```
## Files

- `main.py` :- This is the main code to train the SRGAN model and generate weights for the generator and discriminator. It also generates statistics for our model at each epoch 
- `data_utils.py` :- It's functionality is to preprocess the images and create training and validation datasets from the `data/trainHR` and `data/valHR` directories
- `model.py` :- Contains model architecture (Generator and Discriminator Networks)
- `loss.py` :- It returns the Generator loss required for training using features extracted by VGG model 



## Usage

### Train
```
python main.py
```
- The weights generated will be saved in the `epochs` directory.
- The statistics generated will be saved in the `statistics` directory.

### Upscale Any Image
```
python UpscaleAnyImage/UpscaleAnyImage.py
```
- First train the model and then take the generated weights for the generator network and store it in the `UpscaleAnyImage/TrainedModel` directory
- Then store the low resolution images you want to upscale in the `UpscaleAnyImage/InputImages` directory and run the above code
- The upscaled images will be saved in the `UpscaleAnyImage/UpscaledImages` directory

### Upscale Faces
```
python UpscaleFaces/UpscaleFaces.py
```
- First train the model using [img_align_celeba](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and then take the generated weights for the generator network and store it in the `UpscaleFaces/TrainedModel` directory
- Then store the low resolution group photo you want to upscale in the `UpscaleFaces/InputImage` directory and run the above code
- Using CAFFE model the faces will be detected, cropped and saved in the `UpscaleFaces/DetectedFaces` directory
- Individual faces detected by the CAFFE model will be upscaled and saved in the `UpscaleFaces/UpscaledFaces` directory

