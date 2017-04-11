import numpy as np
from matplotlib import pyplot as plt
from skimage import data, io, color
from skimage.transform import integral_image
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches

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

# Загрузка изображения
image = io.imread("images/002.jpg")
# Перевод в полутоновый формат
image_halftone = color.rgb2gray(image)
# Специальный формат изображения для нахождения признаков
im = integral_image(image_halftone)

# Матрицы для нахождения сохраненых признаков
matrix = np.ndarray(shape=(len(im), len(im[0])))
matrix_bool = np.ndarray(shape=(len(im), len(im[0])), dtype=bool)

# Получение признаков Хаара
for i in range(len(im) - 5):
    for j in range(len(im[i]) - 5):
        # получили матрицу значений Хаара
        matrix[i, j] = haar_feature(im, i, j, 0, 1)

# Фильтр для признаков Хаара
filter_number_min = 50
filter_number_max = 80

# Применение фильтра к вычесленным значениям
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if ((((matrix[i][j])*1000) > filter_number_min) and (((matrix[i][j])*1000) < filter_number_max)):
            matrix_bool[i, j] = True
        else:
            matrix_bool[i, j] = False

# Создание слоя для отметок
label_image = label(matrix_bool)
# Наслойка отметок на картинку с фильтром
image_label_overlay = label2rgb(label_image, image=image_halftone)
# Создает дополнительную фигуру
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

coord = {}
i = 1

# В цикде проходтимся по всем местам, где есть отметки
for region in regionprops(label_image):

    # Получаем регионы с доcтаточным количеством отметок,
    # где отметка больше определенного размера
    if region.area >= 5:

        # Рисуем на них прямоугольники
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(rect)

        y0, x0 = region.centroid

        # для маркера
        ax.plot(x0, y0, 'rs', markersize=10)
        ax.text(x0, y0, str(i), fontsize=12)

        print("[%s] x0: %s y0: %s\n" % (i, x0, y0,))
        i += 1

ax.set_axis_off()
plt.tight_layout()
plt.show()