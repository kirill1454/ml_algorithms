import pandas as pd
import numpy as np
from functools import reduce
from operator import add


class MyLineReg:
    def __init__(self, weights=[], n_iter=100, learning_rate=0.1):
        """
        Функция инициализирует объект класса

        :param weights: Изначальные веса модели
        :param n_iter: Количество циклов градиентного спуска
        :param learning_rate: Шаг градиентного спуска
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, x_train, y_train, verbose=False):
        """
        Функция обучает модель, функция потерь MSE, минимум функции
        определяется градиентным спуском

        :param x_train: Входные признаки
        :param y_train: Целевой признак
        :param verbose: Вывод лога обучения
        :return: None
        """
        x_train.insert(0, 'col_ones', 1)
        n = x_train.shape[1]
        x_train = x_train.to_numpy()
        self.weights = [1 for _ in range(n)]
        for i in range(self.n_iter):
            y_pred = np.dot(x_train, self.weights)
            residuals = list(map(lambda x, y: x - y, y_pred, y_train))
            loss = (1 / x_train.shape[0] *
                    reduce(add, list(map(lambda x, y: (x - y) ** 2, y_pred, y_train))))
            grad_mse = 2 / x_train.shape[0] * np.dot(residuals, x_train)               # градиент MSE
            self.weights = self.weights - self.learning_rate * grad_mse    # шаг в сторону антиградиента

            # вывод лога обучения модели
            if verbose:
                if i == 0:
                    print(f'start | loss: {loss}')
                elif (i + 1) % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss}')

    def get_coef(self):
        """
        Функция возвращает веса обученной модели.

        :return: Список весов
        """
        return self.weights[1:]

    def predict(self, x_test):
        """
        Функция делает предсказание на новых данных

        :param x_test: Тестовая выборка
        :return:
        """
        x_test.insert(0, 'col_ones', 1)
        x_test = x_test.to_numpy()
        y_pred = np.dot(x_test, self.weights)
        return y_pred

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
