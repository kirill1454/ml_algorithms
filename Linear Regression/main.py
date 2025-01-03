import pandas as pd
import numpy as np
from math import sqrt
from typing import Callable


class MyLineReg:
    def __init__(self,
                 weights: list = [],
                 n_iter: int = 100,
                 learning_rate: float | Callable = 0.1,
                 metric: str = 'mse',
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0.3):
        """
        Функция инициализирует объект класса

        :param weights: Изначальные веса модели
        :param n_iter: Количество циклов градиентного спуска
        :param learning_rate: Шаг градиентного спуска
        """
        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = weights

        self.y_train = None
        self.y_pred = None
        self.n_rows = None

        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        if metric not in metrics:
            raise ValueError(f'Выберите метрику из следующего списка: {metrics}')
        else:
            self.metric = metric

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def fit(self, x_train, y_train, verbose=False) -> None:
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

        for i in range(1, self.n_iter + 1):
            y_pred = np.dot(x_train, self.weights)
            residuals = y_pred - y_train

            # Вычисление функции потерь
            mse = 1 / n_rows * sum((y_train - y_pred)**2)
            lasso = self.l1_coef * sum([abs(weight) for weight in self.weights])
            ridge = self.l2_coef * sum([weight ** 2 for weight in self.weights])

            losses = {None: mse,
                      'l1': mse + lasso,
                      'l2': mse + ridge,
                      'elasticnet': mse + lasso + ridge}

            loss = losses[self.reg]

            # Вычисление градиента функции потерь
            mse_grad = 2 / n_rows * residuals @ x_train
            lasso_grad = self.l1_coef * np.sign(self.weights)
            ridge_grad = 2 * np.array([self.l2_coef]) @ np.array([self.weights])

            losses_grad = {None: mse_grad,
                           'l1': mse_grad + lasso_grad,
                           'l2': mse_grad + ridge_grad,
                           'elasticnet': mse_grad + lasso_grad + ridge_grad}

            loss_grad = losses_grad[self.reg]

            # Шаг в сторону антиградиента
            if not callable(self.lr):
                lr = self.lr
                self.weights = self.weights - lr * loss_grad
            else:
                lr = self.lr(i)
                self.weights = self.weights - lr * loss_grad

            # вывод лога обучения модели
            if verbose and self.metric is not None:

                # Перерасчет предсказаний с учетом новых весов
                y_pred = np.dot(x_train, self.weights)

                metrics = {'mae': 1 / n_rows * sum(abs(y_train - y_pred)),
                           'mse': 1 / n_rows * sum((y_train - y_pred)**2),
                           'rmse': sqrt(1 / n_rows * sum((y_train - y_pred)**2)),
                           'mape': 100 / n_rows * sum(abs(y_train - y_pred) / abs(y_train)),
                           'r2': 1 - sum((y_train - y_pred)**2) / sum((y_train - np.mean(y_train))**2)}

                if i == 1:
                    print(f'start | loss: {loss} | {self.metric}: {metrics[self.metric]}'
                          f' | learning_rate: {lr}')
                elif i % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss} '
                          f'| {self.metric}: {metrics[self.metric]}'
                          f' | learning_rate: {lr}')

        self.y_pred = x_train @ self.weights
        self.y_train = y_train
        self.n_rows = n_rows

    def get_best_score(self) -> float:
        metrics = {'mae': 1 / self.n_rows * sum(abs(self.y_train - self.y_pred)),
                   'mse': 1 / self.n_rows * sum((self.y_train - self.y_pred) ** 2),
                   'rmse': sqrt(1 / self.n_rows * sum((self.y_train - self.y_pred) ** 2)),
                   'mape': 100 / self.n_rows *
                           sum(abs(self.y_train - self.y_pred) / abs(self.y_train)),
                   'r2': 1 - sum((self.y_train - self.y_pred) ** 2)
                         / sum((self.y_train - np.mean(self.y_train)) ** 2)}
        return metrics[self.metric]

    def get_coef(self) -> list[float]:
        return self.weights[1:]

    def predict(self, x_test) -> list[float]:
        x_test.insert(0, 'col_ones', 1)
        x_test = x_test.to_numpy()
        return np.dot(x_test, self.weights)

    def __str__(self) -> str:
        return f'{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
