import pandas as pd
import numpy as np
from math import sqrt


class MyLineReg:
    def __init__(self, weights: list=[], n_iter: int=100, learning_rate: float=0.1, metric: str=None):
        """
        Функция инициализирует объект класса

        :param weights: Изначальные веса модели
        :param n_iter: Количество циклов градиентного спуска
        :param learning_rate: Шаг градиентного спуска
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

        self.y_train = None
        self.y_pred = None
        self.n_rows = None

        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        if metric not in metrics:
            raise ValueError(f'Выберите метрику из следующего списка: {metrics}')
        else:
            self.metric = metric

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
        n_cols = x_train.shape[1]
        n_rows = x_train.shape[0]
        x_train = x_train.to_numpy()
        self.weights = [1 for _ in range(n_cols)]

        for i in range(self.n_iter):
            y_pred = np.dot(x_train, self.weights)
            residuals = y_pred - y_train
            loss = 1 / n_rows * sum((y_train - y_pred)**2)
            grad_mse = 2 / n_rows * np.dot(residuals, x_train)               # градиент MSE
            self.weights = self.weights - self.learning_rate * grad_mse      # шаг в сторону антиградиента

            # вывод лога обучения модели
            if verbose and self.metric is not None:

                # Перерасчет предсказаний с учетом новых весов
                y_pred = np.dot(x_train, self.weights)

                metrics = {'mae': 1 / n_rows * sum(abs(y_train - y_pred)),
                           'mse': 1 / n_rows * sum((y_train - y_pred)**2),
                           'rmse': sqrt(1 / n_rows * sum((y_train - y_pred)**2)),
                           'mape': 100 / n_rows * sum(abs(y_train - y_pred) / abs(y_train)),
                           'r2': 1 - sum((y_train - y_pred)**2) / sum((y_train - np.mean(y_train))**2)}

                if i == 0:
                    print(f'start | loss: {loss} | {self.metric}: {metrics[self.metric]}')
                elif (i + 1) % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss} '
                          f'| {self.metric}: {metrics[self.metric]}')

        self.y_pred = np.dot(x_train, self.weights)
        self.y_train = y_train
        self.n_rows = n_rows

    def get_best_score(self):
        metrics = {'mae': 1 / self.n_rows * sum(abs(self.y_train - self.y_pred)),
                   'mse': 1 / self.n_rows * sum((self.y_train - self.y_pred) ** 2),
                   'rmse': sqrt(1 / self.n_rows * sum((self.y_train - self.y_pred) ** 2)),
                   'mape': 100 / self.n_rows *
                           sum(abs(self.y_train - self.y_pred) / abs(self.y_train)),
                   'r2': 1 - sum((self.y_train - self.y_pred) ** 2)
                         / sum((self.y_train - np.mean(self.y_train)) ** 2)}
        return metrics[self.metric]

    def get_coef(self):
        return self.weights[1:]

    def predict(self, x_test):
        x_test.insert(0, 'col_ones', 1)
        x_test = x_test.to_numpy()
        return np.dot(x_test, self.weights)

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
