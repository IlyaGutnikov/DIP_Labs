from matplotlib import pyplot as plt
import math
import numpy as np
import matplotlib.patches as mpatches
from skimage import data, io, filters, color, feature, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters import roberts, sobel, prewitt

image1 = np.tri(10, 10, 0)
image2 = color.rgb2gray(image1)

edges1 = filters.roberts(image2)
print(edges1.astype(bool))

label_image = label(edges1.astype(bool))
print(label_image)
image_label_overlay = label2rgb(label_image, image=edges1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image1)

coord = {}
i = 1
for region in regionprops(label_image):

    # take regions with large enough areas
    if region.area >= 1:

        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(rect)

        y0, x0 = region.centroid

        ax.plot(x0, y0, 'rs', markersize=10)
        ax.text(x0, y0, str(i), fontsize=12)

        print("[%s] x0: %s y0: %s\n" % (i, x0, y0,))
        i+=1

ax.set_axis_off()
plt.tight_layout()
plt.show()