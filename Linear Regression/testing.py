from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from main import MyLineReg

X, y = make_regression(n_samples=1000,
                       n_features=5,
                       n_informative=2,
                       noise=15,
                       random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

a = MyLineReg()
a.fit(X, y, verbose=False)
print(a.get_coef())



