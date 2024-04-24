#%% Modules
from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.image import imsave
from scipy.sparse import  spdiags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
# from scipy.sparse.linalg import spsolve 

##________
def ToUint8(targetImg):
   maxVal=np.max(targetImg)
   minVal=np.min(targetImg)
   rng=maxVal-minVal   
   newImg= 255*(targetImg-minVal)/rng    # rescales to 0 to 255,  but still a floating point # between 0 and 255
   newImg=np.uint8(newImg);     # Converts the float to uint8 
   return newImg

# def 

# Main script
# get the image from file
img1 = io.imread('sample.png')    # read in image  (it is RGB)
if (np.ndim(img1)==3): 
   if (img1.shape[2]==4) :
     img1=img1[:,:,[0,1,2]]  # 4 channel image to 3 channel
   img1 = color.rgb2gray(img1) # 3 channel to BW
else:
   imgGray=img1


plt.figure(1)
origImg=plt.imshow(img1)
# origImg.set_cmap('gray')

#%% Convert to uint8 and save
img1=ToUint8(img1)
io.imsave('Result\Airport.png', img1)


