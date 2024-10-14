import numpy as np
from keras.datasets import mnist

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Подготовка данных
X_train = X_train / 255.0
X_test = X_test / 255.0


# Создание однослойного перцептрона
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, learning_rate=0.005, epochs=500):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error


# Создание перцептрона
perceptron = Perceptron(784)

# Обучение перцептрона
perceptron.train(X_train.reshape(-1, 784), y_train)


def evaluate(perceptron, X, y):
    correct = 0
    for i in range(len(X)):
        prediction = perceptron.predict(X[i])
        if np.argmax(prediction) == y[i]:
            correct += 1
    accuracy = correct / len(X) * 100
    return accuracy


# Оценка точности на тестовых данных
test_accuracy = evaluate(perceptron, X_test.reshape(-1, 784), y_test)
print(f"Точность на тестовых данных: {test_accuracy:.2f}%")
