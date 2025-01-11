import numpy as np
from sklearn.metrics import roc_auc_score
import random


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1,
                 weights=None,
                 metric='accuracy',
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample=None,
                 random_state=42):

        self.weights = weights
        self.n_iter = n_iter
        self.lr = learning_rate
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

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

        # Атрибуты SGD
        self.rs = random_state
        self.sgd_sample = sgd_sample

    def __str__(self):
        return f'{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.lr}'

    @staticmethod
    def get_metric_score(y_test, y_proba, metric: str):
        """1 - положительный класс, 0 - отрицательный класс"""

        # получение классов (в зависимости от y_proba и threshold)
        threshold = 0.5
        y_pred = [1 if i > threshold else 0 for i in y_proba]

        # количество положительных классов, которые определены как положительные
        tp = sum([1 if i[0] == 1 and i[1] == 1 else 0 for i in zip(y_test, y_pred)])

        # количество отрицательных классов, которые определены как отрицательные
        tn = sum([1 if i[0] == 0 and i[1] == 0 else 0 for i in zip(y_test, y_pred)])

        # количество отрицательных классов, которые определены как положительные (ошибка 1 рода)
        fp = sum([1 if i[0] == 0 and i[1] == 1 else 0 for i in zip(y_test, y_pred)])

        # количество положительных классов, которые определены как отрицательные (ошибка 2 рода)
        fn = sum([1 if i[0] == 1 and i[1] == 0 else 0 for i in zip(y_test, y_pred)])

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        metrics = {'accuracy': (tp + tn) / (tp + tn + fp + fn),
                   'precision': precision,
                   'recall': recall,
                   'f1': 2 * (precision * recall / (precision + recall)),
                   'roc_auc': roc_auc_score(y_test, [round(i, 10) for i in y_proba])}

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

        # Фиксация отсчета алгоритма рандома
        random.seed(self.rs)

        for i in range(1, self.n_iter + 1):

            # Случай SGD
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

            # Предсказания
            self.y_pred = self.x_train @ self.weights

            # Вероятность принадлежности к классу
            self.y_proba = 1 / (1 + np.exp(-self.y_pred))

            # Расчет функции ошибки
            eps = 1e-15  # Число для избежания взятия логарифма нуля
            a = self.y_train @ np.log(self.y_proba + eps)
            b = (1 - self.y_train) @ np.log(1 - self.y_proba + eps)
            log_loss = - 1 / self.n_rows * (a + b)

            lasso = self.l1_coef * sum([abs(weight) for weight in self.weights])
            ridge = self.l2_coef * sum([weight ** 2 for weight in self.weights])

            losses = {None: log_loss,
                      'l1': log_loss + lasso,
                      'l2': log_loss + ridge,
                      'elasticnet': log_loss + lasso + ridge}

            loss = losses[self.reg]

            # Расчет градиента функции ошибки LogLoss
            residuals = self.y_proba - self.y_train
            log_loss_grad = 1 / self.n_rows * residuals @ self.x_train
            lasso_grad = self.l1_coef * np.sign(self.weights)
            ridge_grad = 2 * np.array([self.l2_coef]) @ np.array([self.weights])

            losses_grad = {None: log_loss_grad,
                           'l1': log_loss_grad + lasso_grad,
                           'l2': log_loss_grad + ridge_grad,
                           'elasticnet': log_loss_grad + lasso_grad + ridge_grad}

            loss_grad = losses_grad[self.reg]

            # Шаг в сторону антиградиента
            if not callable(self.lr):
                lr = self.lr
                self.weights = self.weights - lr * loss_grad
            else:
                lr = self.lr(i)
                self.weights = self.weights - lr * loss_grad

            # Лог обучения
            if verbose and self.metric is not None:

                # Перерасчет предсказаний с учетом новых весов
                self.y_pred = self.x_train @ self.weights

                # Расчет метрики качества
                self.score = self.__class__.get_metric_score(y_test=self.y_train,
                                                             y_proba=self.y_proba,
                                                             metric=self.metric)

                if i == 1:
                    print(f'start | loss: {loss} | '
                          f'{self.metric}: {self.score}'
                          f' | learning_rate: {lr}')
                elif i % verbose == 0:
                    print(f'{(i + 1) // verbose * verbose} | loss: {loss} '
                          f'| {self.metric}: {self.score}'
                          f' | learning_rate: {lr}')

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
        y_pred = self.x_train @ self.weights
        y_proba = 1 / (1 + np.exp(-y_pred))
        return self.__class__.get_metric_score(self.y_train, y_proba, self.metric)
