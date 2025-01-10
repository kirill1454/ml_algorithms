import pandas as pd
import numpy as np


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: float = 0.1,
                 weights=None,
                 metric='accuracy ='):

        self.weights = weights
        self.n_iter = n_iter
        self.lr = learning_rate

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        if metric not in metrics:
            raise ValueError(f'Выберите метрику из следующего списка: {metrics}')
        else:
            self.metric = metric

        self.score = None

        self.x_train = None
        self.y_train = None
        self.n_rows = None
        self.y_pred = None
        self.y_proba = None

    def __str__(self):
        return f'{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.lr}'

    @staticmethod
    def get_metric_score(y_test, y_pred, y_proba, metric: str):
        """1 - положительный класс, 0 - отрицательный класс"""

        # количество положительных классов, которые определены как положительные
        tp = sum([1 if i[0] == 1 and i[1] == 1 else 0 for i in zip(y_test, y_pred)])

        # количество отрицательных классов, которые определены как отрицательные
        tn = sum([1 if i[0] == 0 and i[1] == 0 else 0 for i in zip(y_test, y_pred)])

        # количество отрицательных классов, которые определены как положительные (ошибка 1 рода)
        fp = sum([1 if i[0] == 0 and i[1] == 1 else 0 for i in zip(y_test, y_pred)])

        # количество положительных классов, которые определены как отрицательные (ошибка 2 рода)
        fn = sum([1 if i[0] == 1 and i[1] == 0 else 0 for i in zip(y_test, y_pred)])

        # количество наблюдений положительного класса
        pos = sum([i for i in y_pred if i == 1])

        # количество наблюдения отрицательного класса
        neg = sum([1 for i in y_pred if i == 0])

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Расчет значения ROC AUC
        y_proba = [round(i, 10) for i in y_proba]
        df = (pd.DataFrame(data={'y_proba': y_proba, 'y_test': y_test})
              .sort_values(by='y_proba', ascending=False))

        # Создаем новый столбец 'sum', который будет содержать накопительную сумму единиц
        df['sum'] = df['y_test'].cumsum().shift(fill_value=0)

        # Устанавливаем значение 0 для каждой единицы в 'sum'
        df.loc[df['y_test'] == 1, 'sum'] = 0

        df['diff'] = 0.0

        for i in range(len(df)):
            if df.loc[i, 'y_test'] == 0:  # Если y_test равно 0
                # Считаем количество совпадений y_proba для y_test = 1
                count = df.loc[df['y_test'] == 1, 'y_proba'].value_counts().get(df.loc[i, 'y_proba'], 0)
                # Если совпадения найдены, добавляем n * 0.5 в diff
                df.loc[i, 'diff'] = count * 0.5

        df['result'] = df['sum'] - df['diff']
        roc_auc = 1 / (pos * neg) * sum(df['result'])

        metrics = {'accuracy': (tp + tn) / (tp + tn + fp + fn),
                   'precision': precision,
                   'recall': recall,
                   'f1': 2 * (precision * recall / (precision + recall)),
                   'roc_auc': roc_auc}

        return metrics[metric]

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
            if verbose and self.metric is not None:

                # Перерасчет предсказаний с учетом новых весов
                self.y_pred = self.x_train @ self.weights

                # Расчет метрики качества
                threshold = 0.5
                self.score = self.__class__.get_metric_score(y_test=self.y_train,
                                                             y_pred=[1 if i > threshold else 0 for i in self.y_proba],
                                                             y_proba=self.y_proba,
                                                             metric=self.metric)

                if i == 1:
                    print(f'start | loss: {log_loss} '
                          f'{self.metric}: {self.score}')
                elif i % verbose == 0:
                    print(f'{ i // verbose * verbose} | loss: {log_loss} '
                          f'{self.metric}: {self.score}')

    def get_coef(self):
        return self.weights[1:]

    def predict(self, x_test):
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

    def get_best_score(self):
        return self.score
