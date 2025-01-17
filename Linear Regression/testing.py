from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from main import MyLineReg

x_train, y_train = make_regression(n_samples=1000,
                                   n_features=5,
                                   n_informative=2,
                                   noise=15,
                                   random_state=42)
x_train = pd.DataFrame(x_train)
y_train = pd.Series(y_train)
x_train.columns = [f'col_{col}' for col in x_train.columns]


x_test, y_test = make_regression(n_samples=5,
                                   n_features=5,
                                   n_informative=2,
                                   noise=15,
                                   random_state=42)
x_test = pd.DataFrame(x_test)
y_test = pd.Series(y_test)
x_test.columns = [f'col_{col}' for col in x_test.columns]


a = MyLineReg(n_iter=130, learning_rate=0.01, sgd_sample=None, metric='mae')
a.fit(x_train, y_train, verbose=10)
print(a.get_best_score())




