from matplotlib import pyplot as plt
import math
import numpy as np
import matplotlib.patches as mpatches
from skimage import data, io, filters, color, feature, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb


# open the input image
image1 = io.imread("Меню (1).jpg")
#print(image1.shape)

red_channel = image1[:, :, 0]
green_channel = image1[:, :, 1]
blue_channel = image1[:, :, 2]

mask_r_1 = image1[...,0] > 140
mask_r_2 = image1[...,0] < 190
mask_g_1 = image1[...,1] > 60
mask_g_2 = image1[...,1] < 110
mask_b_1 = image1[...,2] > 30
mask_b_2 = image1[...,2] < 55

image_masked = mask_r_1 & mask_r_2 & mask_g_1 & mask_g_2 & mask_b_1 & mask_b_2

label_image = label(image_masked.astype(bool))
print(label_image)
# Наслойка отметок на картинку с фильтром
image_label_overlay = label2rgb(label_image, image=image_masked)
# Создает дополнительную фигуру
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image1)

coord = {}
i = 1

# В цикде проходтимся по всем местам, где есть отметки
for region in regionprops(label_image):

    # Получаем регионы с доcтаточным количеством отметок,
    # где отметка больше определенного размера
    if region.area >= 500:

        # Рисуем на них прямоугольники
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(rect)

        y0, x0 = region.centroid

        # для маркера
        ax.plot(x0, y0, 'rs', markersize=10)
        ax.text(x0, y0, 'Салат с морковью', fontsize=12)

        #print("[%s] x0: %s y0: %s\n" % (i, x0, y0,))
        i += 1

ax.set_axis_off()
plt.tight_layout()
plt.show()