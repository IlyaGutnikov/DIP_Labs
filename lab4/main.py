from pybrain import SigmoidLayer, LinearLayer, TanhLayer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from skimage import io
from pybrain.tools.shortcuts import buildNetwork
import os

def avg_gray(img, win_size, name):
    rows = img.shape[0]
    cols = img.shape[1]

    res = []

    i = 1
    # for each pixel in image with step of size window
    for row in range(0, rows, win_size):
        for col in range(0, cols, win_size):
            r = 0
            g = 0
            b = 0

            # для каждого пикселя в окне
            for px in range(0, win_size):
                for py in range(0, win_size):
                    r += img[row + py, col + px, 0]
                    g += img[row + py, col + px, 1]
                    b += img[row + py, col + px, 2]

            # вычисление среднего
            avg_r = r / (win_size ** 2)
            avg_g = g / (win_size ** 2)
            avg_b = b / (win_size ** 2)

            avg = (avg_r + avg_g + avg_b) / 3

            i += 1


            for px in range(0, win_size):
                for py in range(0, win_size):
                    img[row + py, col + px, :] = (avg, avg, avg)

            res.append(avg)

    io.imsave(os.getcwd() + '/output/avg_'+ name +'.jpg', img)

    return res

# Размер окна
win_size = 100

# Массив со входными данными для того что бы
# проверить реакцию сети на уже известные образцы
test_data = []

# Создали нейронную сеть по заданию
# 100 - количество входных нейронов
# 10 -  количество скрытых слоев
# 2 - количество выходных нейронов
net = buildNetwork(100, 10, 2)

# Создали пакет данных для обучения сети
ds = SupervisedDataSet(100, 2)

# Очистили и сеть и данные
ds.clear()
net.reset()

# Запонили его данными
images1 = io.imread_collection(os.getcwd()+'/input1/*.jpg')
print("Пак изображений Дорнана: " + str(len(images1)))
i = 1
for im in images1:
    avg = avg_gray(im, win_size, "1"+str(i))
    # 1 1 - значит Дорнан
    ds.addSample(avg, (1, 1))
    test_data.append(avg)
    i += 1

images2 = io.imread_collection(os.getcwd()+'/input2/*.jpg')
print("Пак изображений Хидлстоуна: " + str(len(images1)))
i = 1
for im in images1:
    avg = avg_gray(im, win_size, "2"+str(i))
    # 1 1 - значит Хидлстоун
    ds.addSample(avg, (0, 0))
    test_data.append(avg)
    i += 1

# Создали тренера для сети
trainer = BackpropTrainer(net)

# Обучили сеть до сходимости
trainer.trainOnDataset(ds, 100)

# Получили значение тестового образца
image_test = io.imread(os.getcwd() + "/11.jpg")
test_avg = avg_gray(image_test, win_size, "test_img")

# Проверили на уже обученных образцах
j = 1
for i in test_data:
    print(j, net.activate(i))
    j += 1

# Получили значение для этого образца
print("Тестируем на неизвестном для системы изображении Дорнана")
print(net.activate(test_avg))
