<br/>
<h1 align="center">Session 7: Advanced training Concepts
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)

#### Contributors

<p align="center"> <b>Team - 6</b> <p>
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
    
## Problem Statement

* Check this Repo out: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
* You are going to follow the same structure for your Code from now on. So Create:  
    * models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
    * main.py - from Google Colab, now onwards, this is the file that you'll import (along with model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
        1) training and test loops
        2) data splits between test and train
        3) epochs
        4) batch size
        5) which optimizer to run
        6) do we run a scheduler?
    * utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
        1) image transforms.
        2) gradcam.
        3) misclassification code.
        4) tensorboard related stuff.
        5) advanced training policies, etc, etc.  

* Name this main repo something, and don't call it Assignment8. This is what you'll import for all the rest of the assignments. Add a proper readme describing all the files.

### **The assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:**
* Pull your Github code to google colab (don't copy-paste code)
prove that you are following the above structure
that the code in your google collab notebook is NOTHING.. barely anything. 
* There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
* your colab file must:
  * train resnet18 for 20 epochs on the CIFAR10 dataset
  * show loss curves for test and train datasets
  * show a gallery of 10 misclassified images
  * show gradcam output on 10 misclassified images
  
Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬

<br>
<hr>
<br>





