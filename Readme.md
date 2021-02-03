# Generation of satellite alike images with a conditional GAN

Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 edition, authored by:

* [Luisa Fleta](https://www.linkedin.com/in/luisa-fleta-5a010422/)
* [Darío Cortizo](https://www.linkedin.com/in/dariocortizo/)
* [Diego Tascón](https://www.linkedin.com/in/diego-tascon-9139297/)

Advised by professor [Eva Mohedano](https://www.linkedin.com/in/eva-mohedano-261b6889/)

## Table of Contents <a name="toc"></a>

1. [Introduction](#intro)
    1. [Motivation](#motivation)
    2. [Milestones](#milestones)
2. [The data set](#datasets)
3. [Working Environment](#working_env)
4. [General Architecture](#architecture)
5. [Preliminary Tests](#preliminary)
    1. [First steps](#initial)
    2. [Accessing the dataset](#datasetaccess)
    3. [Finding the right parameters](#parameters)
6. [The quest for improving the results](#improvingresults)
7. [Final Tests](#final)
    1. [Baseline Model on the 100k Dataset](#guse)
    2. [Training the Question Channel](#question)
        1. [Word Embedding + LSTM](#lstm)
        2. [GloVe LSTM](#glove)
     3. [Results Summary](#resultssummary)
8. [Result analysis](#results)
    1. [Accuracies by question type (*best accuracies excluding yes/no questions*)](#best)
    2. [Accuracies by question type (*worst accuracies excluding yes/no questions*)](#worst)
    3. [Interesting samples](#interestingsamples)
9. [Conclusions and Lessons Learned](#conclusions)
10. [Next steps](#next_steps)
11. [References](#references)
12. [Additional samples](#samples)

# Introduction <a name="intro"></a>
Generative Adversarial Networks (GANs) were introduced by [Ian Goodfellow et al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets) in 2014. GANs can make up realistic new samples from the distribution of images learned.

Conditional GANs (cGANs) where introduced by [Mehdi Mirza and Simon Osindero](https://arxiv.org/abs/1411.1784) also in 2014. cGANs allow directing the results generated using class labels, a part of data for impainting or data from other modalities. One of the most famous implementations of cGANs is [pix2pix](https://phillipi.github.io/pix2pix/).

In this project we're exploring the possibilities of applying conditional GANs to generate realistic satellite alike images.

## Motivation <a name="motivation"></a>
*To be rewritten*

We have decided to do this project because we considered that being able to answer a question from an image using AI it's 'cool' and, more importantly, it is a project that due to the multimodal approach requires that you must understand two of the most important disciplines in AI-DL: vision and language processing.

In addition it's a relatively new area (papers from 2014, 2015, ...) with plenty of opportunities for improvement and several possible business applications: 
* Image retrieval - Product search in digital catalogs (e.g: Amazon)
* Human-computer interaction (e.g.: Ask to a camera the weather)
* Intelligence Analysis
* Support visually impaired individuals

<p align="right"><a href="#toc">To top</a></p>

## Milestones <a name="milestones"></a>
- Build a base model
- Discuss possible model improvements
- Tune/Improve base model
- Final model

<p align="right"><a href="#toc">To top</a></p>

# The data set <a name="datasets"></a>

The [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) provides in its training set 180 satellite images and their corresponding masks. All the images and masks have a resolution of 5000x5000 and are stored in TIFF format. Satellite images have 3 channels whilst their corresponding masks have only one channel. The masks label the buildings in the area:

![](images/01-vienna1_couple_resized.jpg)

There are also available 180 test images. Since they lack their corresponding mask, they would only be useful if extracting masks from satellite images, which would be the opposite direction of our work.

The training images were taken from 5 cities with different landscapes: Austin, Chicago, Kitsap, Tyrol and Vienna.

The whole training set is 15GB of space, as satellite images are 72MB each one and masks are around 1MB of size. As the model performed some transformations to each file, a pretransformed dataset was generated to accelerate the training process. Details on the procedure can be found in section [Splitting the model](#splitting).

<p align="right"><a href="#toc">To top</a></p>

# Working environment <a name="working_env"></a>
We have developed the project using [Google Colab](https://colab.research.google.com/), which gave us easy and free access to GPUs. We've used both local Colab and Google Drive storage. For some parts, though, we've also used a local python container based on the offical [Docker Hub image](https://hub.docker.com/_/python).

<p ><img src="images/02-collab.jpg" width="200"> <img src="images/02-gdrive.png" width="200"> <img src="images/02-docker-logo.png" width="200"></p>

<p align="right"><a href="#toc">To top</a></p>

# General Architecture <a name="architecture"></a>
We've implemented a pix2pix model using [PyTorch](https://pytorch.org/). Although the creators of pix2pix have a published [PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), as it combines both a CycleGAN and a pix2pix implementation, we started on a simpler one by [mrzhu](https://github.com/mrzhu-cool/pix2pix-pytorch).

The architecture of pix2pix is similar to that described for the original conditional GAN:

<p> <img src="images/03-cGANeschema.png"> </p>

The generator is implemented with a U-Net of ResNet blocks:

<p> <img src="images/03-pix2pix-U-Net-resnets.svg"> </p>

The discriminator is implemented with a PatchGAN, a fully convolutional model which is able to capture more details from the image:

<p> <strong>Image to be provided </strong></p>

<p align="right"><a href="#toc">To top</a></p>

# Preliminary Tests <a name="preliminary"></a>
## First steps <a name="initial"></a>
Our first steps with the chosen implementation were to understand it, compare it with the original pix2pix implementation and prepare a [Colab notebook](01_TrainOriginalImages.ipynb) to test de code. We incorporated access to our Google Drive account and logging to Tensorboard.

## Accessing the dataset <a name="datasetaccess"></a>
Understanding how to access the dataset was crucial. We followed the recommendations from [this article](https://towardsdatascience.com/preparing-tiff-images-for-image-translation-with-pix2pix-f56fa1e937cb) to discover that the TIFF masks had only one channel and their values were held in a [0:255] range, as PNG or JPEG images. Moreover, the picked implementation transformed the images, both satellite ones and masks, to RGB (adding 2 channels to the masks), so we didn't have to treat the dataset in a special way.

After feeling confortable with short trainings (10 or 20 epochs, 20 images) we started to make longer ones (300 epochs, 115 images), and that revealed a weakness of our model: every epoch could take up to 6 minutes. That resulted in 30 hours of training. Soon, Colab started complaining about our abuse of GPU usage. So we had to make something about it.

The dataset class (DatasetFromFolder) read the 72MB satellite image and the 1MB corresponding mask in every epoch, converted to RGB, resized them to 286x286 images, transformed them to a torch tensor, normalized their values, made a 256x256 random crop and a random flip. Most of that work could be done previously and only once. So we made a [script](transform-dataset.py) to pretransform all the masks and images to 286x286 with normalized values and save them to .npy numpy array files. We also adapted the DatasetFromFolder class to read the new files, transform them into torch tensors to random crop and random flip them. A training epoch now lasted only 13 seconds!

<p> <img src="images/04-PretransformImages.png"> </p>

## Finding the right parameters <a name="parameters"></a>
Once we were able to train 900 epochs in up to 3 hours, we started to run different trainings in order to understand the influence of the chosen parameters and find a good combination of them.

Rising the original learning rate of 0.0002 to 0.002 collapsed the training: the generator only produced blanck images.

<p> <img src="images/05-CollapsedPSNR.png"> </p>

Other values tested for the learning rate (0.0001, 0.0003, 0.001) didn't improve the quality of the obtained images. On the hand, we found that the lambda had a bigger influence in the capacity of the model to learn. With the standard lambda of 10, the model losses flattened after few epochs:

<p> <img src="images/05-FlatLosses.png"></p>

On the other hand, larger lambda values of 25, 50 and 100 helped the model to improve the quality of images proportionally to the number of epochs:

<p><img src="images/05-ProgressingLosses.png"></p>
<p><img src="images/05-TestImageEpoch400.png"></p>
<p><img src="images/05-TestImageEpoch900.png"></p>

## The quest for improving the results <a name="improvingresults"></a>
With a LR of 0.0002 and a lambda of 100 we had a good baseline to improve the results. Many options were at hand:

- Splitting the training images would allow the model to learn more detailed information from the satellite pictures: cars, trees, ...
- Using a different content loss, like [VGG Loss](https://paperswithcode.com/method/vgg-loss), to let the model learn a more perceptual similarity generation
- <strong> More to be added </strong>