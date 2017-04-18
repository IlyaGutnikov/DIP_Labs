from skimage import data, io
from skimage.measure import label, regionprops
import os
import numpy as np


def avg_gray(img, win_size, name):
    # input image rows and cols calculating
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

            # for each pixel in window
            for px in range(0, win_size):
                for py in range(0, win_size):
                    r += img[row + py, col + px ,0]
                    g += img[row + py, col + px ,1]
                    b += img[row + py, col + px ,2]

            # compute average (r,g,b) channels
            avg_r = r / (win_size ** 2)
            avg_g = g / (win_size ** 2)
            avg_b = b / (win_size ** 2)

            avg = (avg_r + avg_g + avg_b) / 3

            i += 1

            # tint each region with avg hue
            for px in range(0, win_size):
                for py in range(0, win_size):
                    # img[row + py, col + px, :] = (avg_r, avg_g, avg_b)
                    img[row + py, col + px, :] = (avg, avg, avg)

            res.append(avg)

    io.imsave(os.getcwd() + '/output/avg_'+ name +'.jpg', img)

    return res


# activation functions init
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# class neural network
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):

        # choose activation function
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # weights init
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]
                                ))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
                            1]))-1)*0.25)

    # study
    def fit(self, X, y, learning_rate=0.2, epochs = 20000):

        # input data to matrix format
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # through hidden layers
            for l in range(len(self.weights)):
                hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                a.append(hidden_inputs)

            # error computation
            error = y[i] - a[-1][:-1]
            deltas = [error * self.activation_deriv(a[-1][:-1])]
            l = len(a) - 2

            # The last layer before the output is handled separately because of
            # the lack of bias node in output
            deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            for l in range(len(a) - 3, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1][:-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights) - 1):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta[:, :-1])

            # Handle last layer separately because it doesn't have a bias unit
            i += 1
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    # predict function
    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights)):
            temp = np.ones(a.shape[0] + 1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))
        return a

# Размер окна для сегментации изображения
win_size = 25

# Входные данные
input_data = []

images1 = io.imread_collection(os.getcwd()+'/pool1/*.jpg')
print("Пак изображений МакЭвоя " + str(len(images1)))
i = 1
for im in images1:
    avg = avg_gray(im, win_size, "mc"+str(i))
    input_data.append(avg)
    i += 1

images2 = io.imread_collection(os.getcwd()+'/pool2/*.jpg')
print("Пак изображений Хидлстоуна " + str(len(images2)))
i = 1
for im in images2:
    avg = avg_gray(im, win_size, "th"+str(i))
    input_data.append(avg)
    i += 1

images3 = io.imread_collection(os.getcwd()+'/pool3/*.jpg')
print("Пак изображений Дорнана" + str(len(images3)))
i = 1
for im in images3:
    avg = avg_gray(im, win_size, "jd"+str(i))
    input_data.append(avg)
    i += 1

images4 = io.imread_collection(os.getcwd()+'/pool4/g*.jpg')
print("Пак изображений девушки" + str(len(images4)))
i = 1
for im in images4:
    avg = avg_gray(im, win_size, "g"+str(i))
    input_data.append(avg)
    i += 1


# Тестовое изображение
test = io.imread(os.getcwd() + '/15.JPG')
test1 = avg_gray(test, win_size, "test")

# Пак изображений для поиска
search = []
for xi in input_data:
    search.append(xi)
search.append(test1)

# neural network init
nn = NeuralNetwork([len(input_data[0]), 5, 1], 'tanh')

# Инициализация
y1 = [1 for i in range(len(images1))]
y2 = [1 for i in range(len(images2))]
y3 = [1 for i in range(len(images3))]
y4 = [1 for i in range(len(images4))]

y = y1 + y2 + y3 + y4

# Обучение нейронной сети
nn.fit(input_data, y, epochs = 25000)

# Получение резултатов
j = 1
for i in search:
    pr = round(nn.predict(i)[0])
    if pr == 1:
        res = "Jamie Dornan"
    else: res = "Unknown"
    print(j, nn.predict(i), res)
    j += 1

