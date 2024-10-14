from keras.datasets import mnist
import numpy as np

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Выбранные цифры
digits = [1, 6, 7, 8]


# Функция для выборки образов
def sample_images(X, y, digits, n_samples, n_test_samples=1):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for digit in digits:
        digit_images = X[y == digit].reshape(-1, 28 * 28)  # Преобразование в одномерный массив
        indices = np.arange(digit_images.shape[0])
        np.random.shuffle(indices)

        train_indices = indices[:n_samples]
        test_indices = indices[n_samples:n_samples + n_test_samples]

        train_images.extend(digit_images[train_indices])
        train_labels.extend([digit] * n_samples)

        test_images.extend(digit_images[test_indices])
        test_labels.extend([digit] * n_test_samples)

    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))


# Словарь для сопоставления меток классов с индексами
digit_to_index = {digit: i for i, digit in enumerate(digits)}


# Функция для преобразования меток в one-hot encoding
def labels_to_one_hot(y, digit_to_index):
    one_hot_labels = np.zeros((len(y), len(digit_to_index)))
    for i, label in enumerate(y):
        one_hot_labels[i, digit_to_index[label]] = 1
    return one_hot_labels


# Получение обучающей и тестовой выборок
(train_images, train_labels), (test_images, test_labels) = sample_images(X_train, y_train, digits, 5)

# Преобразование меток в one-hot encoding
y_train_one_hot = labels_to_one_hot(train_labels, digit_to_index)
y_test_one_hot = labels_to_one_hot(test_labels, digit_to_index)



# Класс перцептрона
class SingleLayerPerceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.rand(input_size, num_classes)
        self.bias = np.random.rand(num_classes)

    @staticmethod
    def threshold_activation(x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        activated_output = np.vectorize(SingleLayerPerceptron.threshold_activation)(weighted_sum)
        return activated_output

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y_true in zip(X_train, y_train):
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += learning_rate * np.outer(x, error)
                self.bias += learning_rate * error
            print(f'Эпоха {epoch}, Ошибка: {np.sum(error)}')

    def evaluate(self, X_test, y_test):
        predictions = np.array([self.predict(x) for x in X_test])
        accuracy = np.mean(np.all(predictions == y_test, axis=1))
        print(f'Точность: {accuracy}')


# Создание и обучение перцептрона
perceptron = SingleLayerPerceptron(784, len(digits))
perceptron.train(train_images, y_train_one_hot, epochs=100, learning_rate=0.1)
perceptron.evaluate(test_images, y_test_one_hot)
