# Image Retrieval Using UNet-based Architecture on CIFAR-10

## Overview and Problem Statement
This project implements a UNet-based Auto-encoder capable of producing original images by giving their averaged image as its input. The tasks involved include selecting 1,000 images from the original dataset stratified by their label, creating train and test sets, designing a proper input-output pipeline, defining an architecture to extract important features from input images, and decoding these features into their original images. The loss function, optimizer, and learning rate are modified and tuned to optimize the model's performance.

![](../images/unet.png)

## Data Preparation
The dataset used is CIFAR-10, downloaded from a Github repository, containing images separated into different folders with their labels in jpg format. The dataset consists of 50,000 records in the training set and 10,000 records in the test set, with 10 different classes as target labels. Each input image has a size of 32 * 32 pixels. The data is divided into training and test sets, and a custom dataset is created to randomly choose images from each group and calculate their average image along with the original images.

## Methodology
Initially, an auto-encoder model with shallow layers is developed, consisting of 3 convolution layers in both the encoder and decoder. However, the results were blurry and not distinct, indicating the shallow network's inability to extract essential features. As a solution, a UNet-based model is implemented, which significantly improves the feature extraction process. The model has two decoders to produce the original images based on their averaged image. The Adam optimizer is used with a learning rate of 1e-3, and Mean Squared Error (MSE) is used as the loss function with average reduction.

## Results
The trained model achieves impressive results, with an MSE of 0.0063 on the training data and 0.062 on the test data. Random samples from the training data demonstrate the model's ability to predict original images accurately.

For detailed information and sample images, refer to the [full project documentation](./report.pdf).