import scipy as sp
import copy
from scipy import signal
import numpy as np
# import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

def clusterKeyPoints(keypointList):
    clusters = {}
    for element in keypointList:
        keyRow = int(round(element[0]/30))
        keyCol = int(round(element[1]/30))
        if (keyRow, keyCol) in clusters:
            clusters[(keyRow, keyCol)].append(element)
        else:
            clusters[(keyRow, keyCol)] = [element]
    clusteredKeypointList = []
    for key in clusters:
        dominantResp = [0,0,0,-1]
        for keyEntry in clusters[key]:
            if (keyEntry[3] > dominantResp[3]):
                dominantResp = copy.deepcopy(keyEntry)
        clusteredKeypointList.append(dominantResp)
    return clusteredKeypointList

# def genLoG(sigma, ampFactor):
#     size = 2*(3*sigma)
#     G = ampFactor*(signal.gaussian(size,sigma))
#     LoG = np.gradient(np.gradient(G))

# LoG = genLoG(5, 10)
sigma = 1 # sigma has to be odd
rgbImg = Image.open("butterfly.jpg") # open colour image
greyImg = rgbImg.convert('1') # convert image to black and white
imgArr = np.array(greyImg)
globalKeypointList = []

for k in range(1,20):
    decImgArr = sp.signal.decimate(imgArr, k,axis=0)
    decImgArr = sp.signal.decimate(decImgArr, k,axis=1)
    height,width = decImgArr.shape
    filterRespArr = np.zeros((height,width))
    sp.ndimage.filters.gaussian_laplace(decImgArr, sigma, filterRespArr)

    squaredFilterResp = np.square(filterRespArr)
    # squaredFilterResp = sp.ndimage.filters.gaussian_filter(squaredFilterResp,sigma)
    threshold = 0.1 * squaredFilterResp.max()
    for row in range(0, height):
        for col in range(0,width):
            if (squaredFilterResp[row][col] < threshold):
                squaredFilterResp[row][col] = 0

    partialGradient = np.gradient(squaredFilterResp)



# plt.figure(1)
# plt.subplot(211)
# plt.imshow(imgArr)
# plt.subplot(212)

# plt.imshow(smallerImg)
# plt.show()

# gradient = np.add(partialGradient[0], partialGradient[1])

    gradientsSignX = np.sign(partialGradient[0])
    gradientsSignY = np.sign(partialGradient[1])
    keypointList = []
    # print threshold
    for row in range(0, height):
        for col in range(0,width):
            if ((gradientsSignX[row][col] == 0) and (gradientsSignY[row][col] == 0) 
                and (squaredFilterResp[row][col] > threshold)):
                keypointList.append(list((row, col, k, squaredFilterResp[row][col])))
            elif (( (gradientsSignX[row][col-2] != gradientsSignX[row][col])
             and (gradientsSignY[row-2][col] != gradientsSignY[row][col]) )
            and (squaredFilterResp[row][col] > threshold) ):
                keypointList.append(list((row, col, k, squaredFilterResp[row][col])))

    clusteredKeypointList = clusterKeyPoints(keypointList)
    for keypoint in clusteredKeypointList:
        globalKeypointList.append(keypoint)

fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(imgArr)
for tup in globalKeypointList:
    circ = Circle((tup[1]*tup[2],tup[0]*tup[2]),radius=2*tup[2],facecolor='None', edgecolor='r', lw=1.5)
    ax.add_patch(circ)
plt.show()
