from matplotlib import pyplot as plt
import math
import numpy as np
import matplotlib.patches as mpatches
from skimage import data, io, filters, color, feature, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb

#image = np.tri(10, 10, 0)
# Загрузка цифрового изображения
image = io.imread("images/5.png")
image_halftone = color.rgb2gray(image)

# Обработка фильтром
edges = filters.roberts(image_halftone)
print(edges.astype(bool))

# Отметка интересеющих областей
label_image = label(edges.astype(bool))
print(label_image)
# Наслойка отметок на картинку с фильтром
image_label_overlay = label2rgb(label_image, image=edges)
# Создает дополнительную фигуру
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

coord = {}
i = 1

# В цикде проходтимся по всем местам, где есть отметки
for region in regionprops(label_image):

    # Получаем регионы с дотаточным количеством отметок
    if region.area >= 500:

        # Рисуем на них прямоугольники
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(rect)

        y0, x0 = region.centroid

        ax.plot(x0, y0, 'rs', markersize=10)
        ax.text(x0, y0, str(i), fontsize=12)

        print("[%s] x0: %s y0: %s\n" % (i, x0, y0,))
        i += 1

ax.set_axis_off()
plt.tight_layout()
plt.show()