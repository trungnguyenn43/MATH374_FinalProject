#%% Modules
from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.image import imsave
from scipy.sparse import  spdiags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
# from scipy.sparse.linalg import spsolve 


# mean filter function 
def meanFilter(myImg,boxRadius=1):
    imgDim=myImg.shape   # get image dimensions 
    ny=imgDim[0]
    nx=imgDim[1]
    newImg=np.zeros([ny,nx])  # initialize filtered image 
    
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
                        total += myImg[y_surround][x_surround]
            
            newImg[y][x] = total/element_count
        
    maxVal=np.max(newImg)
    minVal=np.min(newImg)
    rng=maxVal-minVal   
    newImg= 255*(newImg-minVal)/rng    # rescales to 0 to 255,  but still a floating point # between 0 and 255
    newImg=np.uint8(newImg);     # Converts the float to uint8 
    
    return newImg  # wrong answer!

# median filter function
def medianFilter(myImg,boxRadius=1):
    imgDim=myImg.shape   # get image dimensions 
    ny=imgDim[0]
    nx=imgDim[1]
    newImg=np.zeros([ny,nx])  # initialize filtered image 
    
    for y in range (0, ny): #current Y coordinate
        for x in range (0, nx): #current X coordinate
            element_count = 0    
            median = 0
            SurroundList = []
        
            for y_surround in range(y - boxRadius, y + boxRadius +1):
                if y_surround < 0 or y_surround > ny -1:
                    continue
                
                for x_surround in range(x - boxRadius, x + boxRadius +1):
                    if x_surround < 0 or x_surround > nx -1:
                        continue
                    else:
                        SurroundList.append(myImg[y_surround][x_surround])
                        element_count += 1
            
            SurroundList.sort()
            
            if (element_count % 2) == 1:
                median = SurroundList[int((element_count/2))]
                newImg[y][x] = median
            else:
                mid_element = int(element_count/2) - 1
                median = (SurroundList[mid_element] + SurroundList[mid_element+1])/2
                newImg[y][x] = median    
                
    maxVal=np.max(newImg)
    minVal=np.min(newImg)
    rng=maxVal-minVal   
    newImg= 255*(newImg-minVal)/rng    # rescales to 0 to 255,  but still a floating point # between 0 and 255
    newImg=np.uint8(newImg);     # Converts the float to uint8
             
    return newImg  # wrong answer!


# TV denoised (square norm version)
def TVFilter(myImg,tau,maxIters=10_000):
    print('TV Filter: Creating 2D Sparse Laplacian')
    imgDim=myImg.shape   # get image dimensions 
    ny=imgDim[0]
    nx=imgDim[1]
    vy=np.ones(ny); vx=np.ones(nx);   
    vy0=2*vy; vy0[0]=1; vy0[ny-1]=1; 
    vx0=2*vx; vx0[0]=1; vx0[nx-1]=1;
    diagVals = np.array([vy0,-vy,-vy])
    Ly = spdiags(data = diagVals,diags= [0,-1,1],m=ny, n=ny);  
    diagVals = np.array([vx0,-vx,-vx])
    Lx = spdiags(data = diagVals,diags= [0,-1,1],m=nx, n=nx)
     # denoising matrix:   A = L + tau*I  , where L is 2D Laplacian 
#     L is sum of Kroncker products of 1D Laplacians with identities
    A = spkron(speye(ny),Lx)  +  spkron(Ly,speye(nx))  + tau*speye(nx*ny) 
    # convert original noisy image to 1D array 
    b = myImg.flatten(); 
    print('TV Filter: Laplcian Construction Complete!')
   
    
    tol=1e-3; 
    # initialguess: u=0 vector
    u=np.zeros(ny*nx)
    r= A@u - b;  # compute initial residual
    p=-r;   # first direction
    rSquared=np.dot(r,r); # dot product
    rNorm=np.sqrt(rSquared)  # initial norm of residual  
    k=0  # iteration 0
    # 
    # 
    # INSERT CG LOOP HERE 
    #   
    #
    #
    #
    while rNorm >= tol and k < maxIters:
        t = A * p
        alpha = rSquared/np.dot(t,p)
        uNew = u + alpha * p # new estimate for solution
        rNew = r + alpha * t # new residual
        rSquaredNew = np.dot(rNew, rNew)
        Beta = rSquaredNew / rSquared
        pNew = -rNew + Beta * p # new direction
        p = pNew; u = uNew; rSquared = rSquaredNew; r = rNew;
        rNorm = np.square(rSquared) 
        k += 1
       # print('Iter', k, ' |r| =', "{:.2e}".format(rNorm))

    
    
    print('TV Filter: Done With CG Loop')
    # 
    # rescale to 0 to 255,  reshape u back to image array
    maxVal=max(u); minVal=min(u); rng=maxVal-minVal
    u= 255*(u-minVal)/rng
    u=np.uint8(u)
    u=u.reshape(ny,nx)
    return u 
    # END TVDenoise function 


