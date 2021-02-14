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
    1. [Modifying the dataset to get more details](#moredetails)
    2. [VGG Loss](#vggloss)
7. [The Google Cloud instance](#gcinstance)
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

Conditional GANs offer multiple applications. Using them with satellite images sounded attractive. Moreover, annotating images, specially aerial ones, is a hard and very time consuming task. A cGAN trained to generate images from forged masks can help increasing the size of aerial datasets.

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
We have developed the project using [Google Colab](https://colab.research.google.com/), which gave us easy and free access to GPUs. We've used both local Colab and Google Drive storage. For some parts, though, we've also used a local python container based on the offical [Docker Hub image](https://hub.docker.com/_/python). We've also used a [Google Cloud](https://cloud.google.com/) Deep Learning VM instance for longer trainings.

<p ><img src="images/02-collab.jpg" width="200"> <img src="images/02-gdrive.png" width="200"> <img src="images/02-docker-logo.png" width="200"> <img src="images/02-googlecloud.jpg" width="200"></p>

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

Other values tested for the learning rate (0.0001, 0.0003, 0.001) didn't improve the quality of the obtained images. That wasn't he case of the lambda. We found that the lambda had a bigger influence in the capacity of the model to learn. With the standard lambda of 10, the model losses flattened after few epochs:

<p> <img src="images/05-FlatLosses.png"></p>

On the other hand, larger lambda values of 25, 50 and 100 helped the model to improve the quality of images proportionally to the number of epochs:

<p><img src="images/05-ProgressingLosses.png"></p>
Ground truth mask and generated image (from the training set) in epoch 400:
<p><img src="images/05-TestImageEpoch400.png"></p>
Epoch 900:
<p><img src="images/05-TestImageEpoch900.png"></p>

Our baseline model generated reasonable decent images with our validation masks:

<div id="baselineresults">
    Generated images:
    <div id="baselinesgenerated">
        <img src="images/NoSplitLR0.0002-Lambda100/Generated-austin29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/Generated-chicago29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/Generated-kitsap29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/Generated-tyrol-w29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/Generated-vienna29.jpeg">
    </div>
    Ground truth satellite images:
    <div id="validationgt">
        <img src="images/NoSplitLR0.0002-Lambda100/OriginalResized-austin29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/OriginalResized-chicago29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/OriginalResized-kitsap29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/OriginalResized-tyrol-w29.jpeg">
        <img src="images/NoSplitLR0.0002-Lambda100/OriginalResized-vienna29.jpeg">
    </div>
</div>

# The quest for improving the results <a name="improvingresults"></a>
With a LR of 0.0002 and a lambda of 100 we had a good baseline to improve the results. Many options were at hand:

- Splitting the training images would allow the model to learn more detailed information from the satellite pictures: cars, trees, ...
- Using a different content loss, like [VGG Loss](https://paperswithcode.com/method/vgg-loss), to let the model learn a more perceptual similarity generation
- Using [instance normalization](https://arxiv.org/abs/1607.08022) instead of batch normalization
- <strong> More to be added </strong>

## Modifying the dataset to get more details <a name="moredetails"></a>
The results obtained in our best trainings were far from detailed. The model is originally conceived to receive and produce 256x256 images and we trained it with resized 286x286 images from the 5.000x5.000 originals. That meant we were reducing by 306 times the original images and masks.

So we made up a new dataset splitting the original images and masks into smaller squares. From a couple of a 5.000x5.000 image and mask, we obtained 25 1.000x1.000 images and masks. When resized to 286x286, they were only 12 times samller. That would allow the model learn more details from the images at the cost of having 25 times more images to process. The new dataset was also already resized and normalized, as explained [before](#datasetaccess).

25 times more images would mean spending 5 minutes and a half for every epoch. That is 9 hours for 100 epochs in Google Colab, way too much for the limits the free platform offers. So we decided to create a Google Cloud instance overcome the usage limits of Colab. You can find more details about the instance in [this section](#gcinstance).

As a first test, we created a dataset splitting masks and images only by 2x2, obtaining 540 training 2.500x2.500 couples, 120 test couples and leaving 60 couples for validation. We spent some time to find the best combination of parameters (data loader threads, batch and test batch sizes). A surprise was awaiting: if we could train on Colab spending 13 seconds per epoch (should be 52 sec/epoch with 540 couples), every epoch lasted 97 seconds in our new shining cloud environment. We made a 900 epoch training anyway, which lasted almost 25 hours at a cost of around 16€.

Although the intermediate results recorded in tensorboard were promising, the validation images generated showed color problems. We generated two sets of images: one feeding the whole mask to produce a single 256x256 image:

<div id="fullmaskwith2x2training">
    Generated images:
    <div id="fullmaskwith2x2traininggenerated">
        <img src="images/Split2x2-fullsize/Generated-austin29.jpeg">
        <img src="images/Split2x2-fullsize/Generated-chicago29.jpeg">
        <img src="images/Split2x2-fullsize/Generated-kitsap29.jpeg">
        <img src="images/Split2x2-fullsize/Generated-tyrol-w29.jpeg">
        <img src="images/Split2x2-fullsize/Generated-vienna29.jpeg">
    </div>
    Ground truth satellite images:
    <div id="fullmaskwith2x2trainingvalidation">
        <img src="images/Split2x2-fullsize/OriginalResized-austin29.jpeg">
        <img src="images/Split2x2-fullsize/OriginalResized-chicago29.jpeg">
        <img src="images/Split2x2-fullsize/OriginalResized-kitsap29.jpeg">
        <img src="images/Split2x2-fullsize/OriginalResized-tyrol-w29.jpeg">
        <img src="images/Split2x2-fullsize/OriginalResized-vienna29.jpeg">
    </div>
</div>

In the second set with split the mask into 4 256x256 tiles and thus we generated 4 256x256 splits joined afterwards:

<div id="2x2maskwith2x2training">
    Generated images:
    <div id="2x2maskwith2x2traininggenerated">
        <img src="images/Split2x2-2x2/Generated-austin29.jpeg">
        <img src="images/Split2x2-2x2/Generated-chicago29.jpeg">
        <img src="images/Split2x2-2x2/Generated-kitsap29.jpeg">
        <img src="images/Split2x2-2x2/Generated-tyrol-w29.jpeg">
        <img src="images/Split2x2-2x2/Generated-vienna29.jpeg">
    </div>
    Ground truth satellite images:
    <div id="2x2maskwith2x2trainingvalidation">
        <img src="images/Split2x2-2x2/OriginalResized-austin29.jpeg">
        <img src="images/Split2x2-2x2/OriginalResized-chicago29.jpeg">
        <img src="images/Split2x2-2x2/OriginalResized-kitsap29.jpeg">
        <img src="images/Split2x2-2x2/OriginalResized-tyrol-w29.jpeg">
        <img src="images/Split2x2-2x2/OriginalResized-vienna29.jpeg">
    </div>
</div>

We also made another training with a slightly different approach. We trained our already trained baseline model with the new tiles dataset. *_To be continued..._*


## VGG Loss <a name="vggloss"></a>

Another strategy to improve the quality of the generated images could be replacing our L1 content loss by the [VGG loss](https://paperswithcode.com/method/vgg-loss).

To calculate the VGG loss, a VGG model pretrained on ImageNet classification is used. In the generator training phase, the L1 loss used to compare the generated satellite image and the ground truth satellite image is substituted by a comparison between the classification labels issued by the VGG model between the same both satellite images (generated and GT). The inctuition behind this is that if both images are similar, the labels and confidence scores resulting from the inference of the VGG network will also be similar.

The first results didn't improve apparently those from our baseline training.

*_To be continued..._*
