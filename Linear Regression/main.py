import pandas as pd
import numpy as np
from functools import reduce
from operator import add


class MyLineReg:
    def __init__(self, weights=[], n_iter=100, learning_rate=0.1):
        '''
        n_iter - количество шагов градиентного спуска
        learning_rate - коэффициент скорости обучения градиентного спуска
        weights - веса модели
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, X, y_true, verbose=False):
        '''X - все входные признаки (pd.DataFrame)
           y - таргет (pd.Series),
           verbose - на какой итерации выводить лог'''
        X.insert(0, 'col_ones', 1)
        n = X.shape[1]
        X = X.to_numpy()
        self.weights = [1 for _ in range(n)]
        for i in range(self.n_iter):
            y_pred = np.dot(X, self.weights)
            residuals = list(map(lambda x, y: x - y, y_pred, y_true))
            loss = (1 / X.shape[0] *
                    reduce(add, list(map(lambda x, y: (x - y) ** 2, y_pred, y_true))))
            grad_mse = 2 / X.shape[0] * np.dot(residuals, X)               # градиент MSE
            self.weights = self.weights - self.learning_rate * grad_mse    # шаг в сторону антиградиента

            # вывод лога обучения модели
            if verbose:
                if i == 0:
                    print(f'start | loss: {loss}')
                elif (i + 1) % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss}')

    def get_coef(self):
        return self.weights[1:]

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
