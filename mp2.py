import scipy as sp
import copy
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
sigma = 7 # sigma has to be odd
rgbImg = Image.open("butterfly.jpg") # open colour image
width,height=rgbImg.size
greyImg = rgbImg.convert('1') # convert image to black and white
imgArr = np.array(greyImg)
filterRespArr = np.zeros((height,width))
sp.ndimage.filters.gaussian_laplace(imgArr, sigma, filterRespArr)

squaredFilterResp = np.square(filterRespArr)
# squaredFilterResp = sp.ndimage.filters.gaussian_filter(squaredFilterResp,sigma)
threshold = 0.4 * squaredFilterResp.max()
for row in range(0, height):
    for col in range(0,width):
        if (squaredFilterResp[row][col] < threshold):
            squaredFilterResp[row][col] = 0

partialGradient = np.gradient(squaredFilterResp)

# plt.figure(1)
# plt.subplot(211)
# plt.imshow(squaredFilterResp)
# plt.subplot(212)
# plt.imshow(partialGradient[0])
# plt.show()

# gradient = np.add(partialGradient[0], partialGradient[1])

gradientsSignX = np.sign(partialGradient[0])
gradientsSignY = np.sign(partialGradient[1])

keypointList = []

# print threshold
for row in range(0, height):
    for col in range(0,width):
        # print gradientSign[row][col] 
        if ((gradientsSignX[row][col] == 0) and (gradientsSignY[row][col] == 0) 
            and (squaredFilterResp[row][col] > threshold)):
            keypointList.append(list((row, col, sigma, squaredFilterResp[row][col])))
        elif (( (gradientsSignX[row][col-2] != gradientsSignX[row][col])
         and (gradientsSignY[row-2][col] != gradientsSignY[row][col]) )
        and (squaredFilterResp[row][col] > threshold) ):
            keypointList.append(list((row, col, sigma, squaredFilterResp[row][col])))

clusters = {}

for element in keypointList:
    # print("before:", str(element[0])+","+str(element[0]))
    keyRow = int(round(element[0]/10))
    keyCol = int(round(element[1]/10))
    if (keyRow, keyCol) in clusters:
        clusters[(keyRow, keyCol)].append(element)
    else:
        clusters[(keyRow, keyCol)] = [element]
    # element[0] = int(round(element[0]/10))
    # element[1] = int(round(element[1]/10))
    # element[0] = element[0] * 10
    # element[1] = element[1] * 10
    # print("after:", str(element[0])+","+str(element[0]))
clusteredKeypointList = []
for key in clusters:
    dominantResp = [0,0,0,-1]
    for keyEntry in clusters[key]:
        if (keyEntry[3] > dominantResp[3]):
            dominantResp = copy.deepcopy(keyEntry)
    clusteredKeypointList.append(dominantResp)

# print clusters

# print keypointList
# print compressed

fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(imgArr)
for tup in clusteredKeypointList:
    circ = Circle((tup[1],tup[0]),radius=tup[2],facecolor='None', edgecolor='r', lw=1.5)
    ax.add_patch(circ)
plt.show()

# plt.figure(1)
# plt.subplot(211)
# plt.plot(squaredFilterResp[200])
# plt.subplot(212)
# plt.plot(gradient[200])
# plt.show()