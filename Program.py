#% Modules
from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from scipy.sparse import  spdiags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron

from RoadDetect import *
# from scipy.sparse.linalg import spsolve 

def DisplayImage(img, num):
   plt.figure(num)
   plt.imshow(img)
   plt.axis('off')
   plt.show()
   return num+1

def SaveImg(img, name):
   newImg=np.uint8(img)
   io.imsave(name, newImg)


def ImageInput(path):
   img = io.imread(path)
   if (np.ndim(img)==3): 
      if (img.shape[2]==4) :
         img=img[:,:,[0,1,2]]  # 4 channel image to 3 channel
      else:
         imgGray=img
         return imgGray
   return img


# =================================
# Main script
# get the image from file
figureOrder = 1

img1 = ImageInput(path='sample3.png')  # read in image  (it is RGB)

# img1 = meanFilter(img1, 3)

# output = RoadDetect(img1, 0.85)

figureOrder = DisplayImage(img1, figureOrder)

R, G, B = ChannelSplit(img1)

#%%
RG = R + G
RB = R + B
GB = G + B

RRG = RG + R

figureOrder = DisplayImage(RRG, figureOrder)

# SaveImg(output, 'Result\Airport.png')




