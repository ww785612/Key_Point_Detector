import scipy as sp
from scipy import signal
import numpy as np
# import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

# def genLoG(sigma, ampFactor):
#     size = 2*(3*sigma)
#     G = ampFactor*(signal.gaussian(size,sigma))
#     LoG = np.gradient(np.gradient(G))

# LoG = genLoG(5, 10)
sigma = 11 # sigma has to be odd
rgbImg = Image.open("butterfly.jpg") # open colour image
width,height=rgbImg.size
greyImg = rgbImg.convert('1') # convert image to black and white
imgArr = np.array(greyImg)
filterRespArr = np.zeros((height,width))
sp.ndimage.filters.gaussian_laplace(imgArr, sigma, filterRespArr)

# plt.figure(1)
# plt.subplot(211)
# squaredFilterResp = np.square(filterRespArr)
# plt.imshow(squaredFilterResp)
# plt.subplot(212)
# squaredFilterResp2 = sp.ndimage.filters.gaussian_filter(squaredFilterResp,sigma)
# plt.imshow(squaredFilterResp2)
# plt.show()

squaredFilterResp = np.square(filterRespArr)
# squaredFilterResp = sp.ndimage.filters.gaussian_filter(squaredFilterResp,sigma)

partialGradient = np.gradient(squaredFilterResp)

# gradient = np.add(partialGradient[0], partialGradient[1])

gradientsSignX = np.sign(partialGradient[0])
gradientsSignY = np.sign(partialGradient[1])

keypointList = []
threshold = 0.3 * squaredFilterResp.max()

for row in range(0, height):
    for col in range(0,width):
        # print gradientSign[row][col] 
        if ((gradientsSignX[row][col] == 0) and (gradientsSignY[row][col] == 0) 
            and (squaredFilterResp[row][col] > threshold)):
            keypointList.append((row, col, sigma))
        elif (( (gradientsSignX[row][col-1] != gradientsSignX[row][col])
         and (gradientsSignY[row-1][col] != gradientsSignY[row][col]) )
        and (squaredFilterResp[row][col] > threshold) ):
            keypointList.append((row, col, sigma))

# print keypointList
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(imgArr)
for tup in keypointList:
    circ = Circle((tup[1],tup[0]),radius=tup[2],facecolor='None', edgecolor='r', lw=1.5)
    ax.add_patch(circ)
plt.show()

# plt.figure(1)
# plt.subplot(211)
# plt.plot(squaredFilterResp[200])
# plt.subplot(212)
# plt.plot(gradient[200])
# plt.show()