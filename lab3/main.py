import numpy as np
from skimage import data, io, color
from skimage.transform import integral_image
import collections

def haar_feature(i, x, y, f, s):

    """
    Haar Features
    It returns the specific haar feature, evaluated in a 24*24 frame,
    as a single array.

    Input
    -----
    i : integral image
    x : the x-co-ordinate
    y : the y-co-ordinate
    f : feature type
    s : scale factor

    Output
    ------
    haar_features = computed haar value
    """
    features = np.array([[2, 1], [1, 2], [3, 1], [1, 3], [2, 2]])
    h = features[f][0]*s
    w = features[f][1]*s

    if f == 0:
        bright = (i[int(x+h/2-1), y+w-1] + i[x-1, y-1]) - (i[x-1, y+w-1] + i[int(x+h/2-1), y-1])
        dark = (i[x+h-1, y+w-1] + i[int(x+h/2-1), y-1]) - (i[int(x+h/2-1), y+w-1] + i[x+h-1, y-1])
    elif f == 1:
        bright = (i[x+h-1, int(y+w/2-1)] + i[x-1, y-1]) - (i[x-1, int(y+w/2-1)] + i[x+h-1, y-1])
        dark = (i[x+h-1, y+w-1] + i[x-1, int(y+w/2-1)]) - (i[x+h-1, int(y+w/2-1)] + i[x-1, y+w-1])
    #print(bright)
    #print(dark)
    haar_feature_val = bright-dark
    #print(haar_feature_val)
    return haar_feature_val

image = io.imread("images/001.jpg")

image_halftone = color.rgb2gray(image)

im = integral_image(image_halftone)

#print(im)

Matrix = collections.defaultdict(float)

haar_feature(im, 995, 995, 0, 1)

for i in range(len(im) - 5):
    for j in range(len(im[i]) - 5):
        Matrix[i, j] = haar_feature(im, i, j, 0, 1)

print(Matrix)

io.imshow(image)