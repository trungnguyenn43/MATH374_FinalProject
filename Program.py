#% Modules
from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.image import imsave
from scipy.sparse import  spdiags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
# from scipy.sparse.linalg import spsolve 

##________


# ToUint8: Converts a floating point image to a uint8 image -> 0 to 255
def ToUint8(targetImg):
   maxVal=np.max(targetImg)
   minVal=np.min(targetImg)
   rng=maxVal-minVal
   # rescales to 0 to 255,  but still a floating point # between 0 and 255
   newImg= 255*(targetImg-minVal)/rng
   # Converts the float to uint8
   newImg=np.uint8(newImg)
   return newImg

def GreenDropper(img):
      for pixel in range(img.shape[1]):
         img[:,:,[1]]=0
      return img

def DisplayImage(img, num):
   plt.figure(num)
   plt.imshow(img)
   plt.axis('off')
   plt.show()
   num=num+1

def SaveImg(img, name):
   ## Convert to uint8 and save
   img=ToUint8(img)
   io.imsave(name, img)

# Main script
# get the image from file
img1 = io.imread('sample.png')    # read in image  (it is RGB)
if (np.ndim(img1)==3): 
   if (img1.shape[2]==4) :
     img1=img1[:,:,[0,1,2]]  # 4 channel image to 3 channel
   # img1=GreenDropper(img1)
else:
   imgGray=img1


figureOrder = 1
DisplayImage(img1, figureOrder)
SaveImg(img1, 'Result\Airport.png')

