# How to run the code
The denoising algorithm is a frequency based method that follows human visual characteristics. This README file will serve as a step-by-step guide on how to run the code 
## Pre-requisites
Ensure you have Python installed. Install the required packages using:

```sh
pip install numpy scipy pillow opencv-python matplotlib scikit-image

```
## Run the following imports

``` sh
import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
from PIL import Image
import itertools
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os
```
