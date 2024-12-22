import pandas as pd


class MyLineReg:
    def __init__(self, weight, n_iter=100, learning_rate=0.1):
        '''
        n_iter - количество шагов градиентного спуска
        learning_rate - коэффициент скорости обучения градиентного спуска
        weights - веса модели
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y, verbose=False):
        '''X - все входные признаки (pd.DataFrame)
           y - таргет (pd.Series),
           verbose - на какой итерации выводить лог'''


    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

a = MyLineReg()
print(a)