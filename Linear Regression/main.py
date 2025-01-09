import pandas as pd
import numpy as np
from math import sqrt
from typing import Callable
import random


class LinearRegression:
    def __init__(self,
                 weights: list = [],
                 n_iter: int = 100,
                 learning_rate: float | Callable = 0.1,
                 metric: str = 'mse',
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0.3,
                 sgd_sample: float = None,
                 random_state: int = 42):
        """
        Инициализирует объект класса

        :param weights: Веса модели
        :param n_iter: Итерации градиентного спуска
        :param learning_rate: Шаг в сторону антиградиента
        :param metric: Метрика качества
        :param reg: Регуляризация
        :param l1_coef: Коэффициент лассо регуляризации
        :param l2_coef: Коэффициент ридж регуляризации
        :param sgd_sample: Размер мини-батча
        :param random_state: Инициализатор числового генератора
        """
        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = weights

        self.x_train = None
        self.y_train = None
        self.y_pred = None
        self.n_rows = None
        self.score = None

        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        if metric not in metrics:
            raise ValueError(f'Выберите метрику из следующего списка: {metrics}')
        else:
            self.metric = metric

        # Атрибуты регуляризации
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.rs = random_state
        self.sgd_sample = sgd_sample

    @staticmethod
    def get_metric_score(y_train, y_pred, n_rows, metric: str):
        metrics = {'mae': 1 / n_rows * sum(abs(y_train - y_pred)),
                   'mse': 1 / n_rows * sum((y_train - y_pred) ** 2),
                   'rmse': sqrt(1 / n_rows * sum((y_train - y_pred) ** 2)),
                   'mape': 100 / n_rows * sum(abs(y_train - y_pred) / abs(y_train)),
                   'r2': 1 - sum((y_train - y_pred) ** 2) / sum((y_train - np.mean(y_train)) ** 2)}
        return metrics[metric]

    def fit(self, x_train, y_train, verbose=False) -> None:
        """
        Обучает модель, функция потерь MSE, минимум функции
        определяется градиентным спуском

        :param x_train: Входные признаки
        :param y_train: Целевой признак
        :param verbose: Вывод лога обучения
        :return: None
        """
        # Добавление фиктивного столбца
        x_train.insert(0, 'col_ones', 1)

        # Количество строк/столбцов датасета
        n_cols = x_train.shape[1]
        n_rows = x_train.shape[0]

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        # Изначальные веса
        self.weights = [1 for _ in range(n_cols)]

        random.seed(self.rs)

        for i in range(1, self.n_iter + 1):

            # Случай стохастического градиентного спуска
            if self.sgd_sample is not None:
                sample_rows_idx = (random.sample(range(n_rows), self.sgd_sample) if isinstance(self.sgd_sample, int)
                                   else random.sample(range(n_rows), int(self.sgd_sample * n_rows)))
                self.x_train = x_train[sample_rows_idx]
                self.y_train = y_train[sample_rows_idx]
                self.n_rows = len(sample_rows_idx)
            else:
                self.x_train = x_train
                self.y_train = y_train
                self.n_rows = n_rows

            self.y_pred = np.dot(self.x_train, self.weights)
            residuals = self.y_pred - self.y_train

            # Вычисление функции потерь
            mse = 1 / self.n_rows * sum((self.y_train - self.y_pred)**2)
            lasso = self.l1_coef * sum([abs(weight) for weight in self.weights])
            ridge = self.l2_coef * sum([weight ** 2 for weight in self.weights])

            losses = {None: mse,
                      'l1': mse + lasso,
                      'l2': mse + ridge,
                      'elasticnet': mse + lasso + ridge}

            loss = losses[self.reg]

            # Вычисление градиента функции потерь
            mse_grad = 2 / self.n_rows * residuals @ self.x_train
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

                # Расчет метрики качества
                self.score = self.__class__.get_metric_score(y_train, y_pred, n_rows, self.metric)

                if i == 1:
                    print(f'start | loss: {loss} | '
                          f'{self.metric}: {self.score}'
                          f' | learning_rate: {lr}')
                elif i % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss} '
                          f'| {self.metric}: {self.score}'
                          f' | learning_rate: {lr}')

        self.y_pred = x_train @ self.weights
        self.y_train = y_train
        self.n_rows = n_rows

    def get_best_score(self) -> float:
        return self.score

    def get_coef(self) -> list[float]:
        return self.weights[1:]

    def predict(self, x_test) -> list[float]:
        x_test.insert(0, 'col_ones', 1)
        x_test = x_test.to_numpy()
        return np.dot(x_test, self.weights)

    def __str__(self) -> str:
        return f'{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
