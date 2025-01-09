import pandas as pd
import numpy as np

class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: float = 0.1,
                 weights=None):

        self.weights = weights
        self.n_iter = n_iter
        self.lr = learning_rate

        self.x_train = None
        self.y_train = None
        self.n_rows = None
        self.y_pred = None
        self.y_proba = None

    def __str__(self):
        return f'{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.lr}'

    def fit(self, x_train, y_train, verbose=False):

        # Добавление фиктивного столбца
        x_train.insert(0, 'col_ones', 1)

        # Количество строк/столбцов датасета
        n_cols = x_train.shape[1]
        n_rows = x_train.shape[0]

        # Преобразование в массивы numpy
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        # Изначальные веса
        self.weights = np.ones(n_cols)

        for i in range(1, self.n_iter + 1):
            self.x_train = x_train
            self.y_train = y_train
            self.n_rows = n_rows

            # Предсказания
            self.y_pred = self.x_train @ self.weights

            # Вероятность принадлежности к классу
            self.y_proba = 1 / (1 + np.exp(-self.y_pred))

            # Расчет функции ошибки LogLoss
            eps = 1e-15 # Число для избежания взятия логарифма нуля
            a = self.y_train @ np.log(self.y_proba + eps)
            b = (1 - self.y_train) @ np.log(1 - self.y_proba + eps)
            log_loss = - 1 / self.n_rows * (a + b)

            # Расчет градиента функции ошибки LogLoss
            residuals = self.y_proba - self.y_train
            log_loss_grad = 1 / self.n_rows * residuals @ x_train

            # Шаг в сторону антиградиента
            self.weights = self.weights - self.lr * log_loss_grad

            # Лог обучения
            if verbose:

                # Перерасчет предсказаний с учетом новых весов
                y_pred = self.x_train @ self.weights

                if i == 1:
                    print(f'start | loss: {log_loss}')
                elif i % verbose == 0:
                    print(f'{ i // verbose * verbose} | loss: {log_loss}')

    def get_coef(self):
        return self.weights[1:]

    def predict(self, x_test) -> list[float]:
        sample = x_test.copy()
        sample.insert(0, 'col_ones', 1)
        sample = sample.to_numpy()

        # порог
        threshold = 0.5

        # вероятности
        y_proba = 1 / (1 + np.exp(-(sample @ self.weights)))

        return [1 if i > threshold else 0 for i in y_proba]

    def predict_proba(self, x_test):
        sample = x_test.copy()
        sample.insert(0, 'col_ones', 1)
        sample = sample.to_numpy()
        return 1 / (1 + np.exp(-(sample @ self.weights)))
