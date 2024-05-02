from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def ToUint8(target):
    maxVal=np.max(target)
    minVal=np.min(target)
    rng=maxVal-minVal   
    newImg= 255*(target-minVal)/rng    # rescales to 0 to 255,  but still a floating point # between 0 and 255
    newImg=np.uint8(newImg);     # Converts the float to uint8 
    return newImg

def ChannelSplit(img):

    imgR= 1 * img
    imgG= 1 * img
    imgB= 1 * img

    imgR[:,:,1] = 0
    imgR[:,:,2] = 0

    imgG[:,:,0] = 0
    imgG[:,:,2] = 0

    imgB[:,:,0] = 0
    imgB[:,:,1] = 0

    
    return imgR, imgG, imgB

def CalMean(img):
    mean = np.mean(img)
    return mean


    
def meanFilter(myImg,boxRadius=1):
    imgDim=myImg.shape   # get image dimensions 
    ny=imgDim[0]
    nx=imgDim[1]
    nz=imgDim[2]
    newImg=np.zeros([ny,nx,nz])  # initialize filtered image 
    
    for z in range(nz):
        for y in range (ny): #current Y coordinate
            for x in range (nx): #current X coordinate
                element_count = 0
                total = 0
                
                for y_surround in range(y - boxRadius, y + boxRadius +1):
                    if y_surround < 0 or y_surround > ny -1:
                        continue
                    
                    for x_surround in range(x - boxRadius, x + boxRadius +1):
                        if x_surround < 0 or x_surround > nx -1:
                            continue
                        else:
                            element_count += 1
                            total += myImg[y_surround][x_surround][z]
                
                newImg[y][x][z]= total/element_count
        
    newImg = ToUint8(newImg)
    
    return newImg  # wrong answer!
    
    