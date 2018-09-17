# Semantic Segmentation using U-Net on Pascal VOC 2012
This repository implements semantic segmentation on Pascal VOC2012 using U-Net.

An article about this implementation is [here](https://qiita.com/tktktks10/items/0f551aea27d2f62ef708).

Semantic segmentation is a kind of image processing as below.

<img src=https://raw.githubusercontent.com/tks10/Images/master/UNet/horse_original.jpeg width=50%><img src=https://raw.githubusercontent.com/tks10/Images/master/UNet/horse_segmented.png width=50%>

This package includes modules of data loader, reporter(creates reports of experiments), data augmenter, u-net model, and training it.

# Usage
To show how to run.

`python main.py --help`


To run with data augmentation using GPUs.

`python main.py --gpu --augmentation`


# U-Net
U-Net is an encoder-decoder model consisted of only convolutions, without fully connected layers.

U-Net has a shape like "U" as below, that's why it is called U-Net.

<img src=https://raw.githubusercontent.com/tks10/Images/master/UNet/unet.png width=75%>


# Experiments

The following results is got by default settings.

## Results of segmentation
### For the training set
![training1](https://raw.githubusercontent.com/tks10/Images/master/UNet/train1.png)

![training2](https://raw.githubusercontent.com/tks10/Images/master/UNet/train2.png)

### For the test set
![test1](https://raw.githubusercontent.com/tks10/Images/master/UNet/test1.png)

![test2](https://raw.githubusercontent.com/tks10/Images/master/UNet/test2.png)

![test3](https://raw.githubusercontent.com/tks10/Images/master/UNet/test3.png)



## Accuracy and Loss
<img src=https://raw.githubusercontent.com/tks10/Images/master/UNet/accuracy.png width=35%> <img src=https://raw.githubusercontent.com/tks10/Images/master/UNet/loss.png width=35%>





