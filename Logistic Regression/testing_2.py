import pandas as pd
from sklearn.metrics import roc_auc_score

y_proba = [0.98, 0.96, 0.88, 0.77, 0.6,  0.6, 0.5, 0.5, 0.5]
y_test = [1, 1, 0, 0, 1, 0, 1, 1, 0]

pos = y_test.count(1)
neg = y_test.count(0)


# Создаем DataFrame и сортируем по вероятностям
df = pd.DataFrame(data={'y_proba': y_proba, 'y_test': y_test}).sort_values(by='y_proba', ascending=False)

# Создаем новый столбец 'sum', который будет содержать накопительную сумму единиц
df['sum'] = df['y_test'].cumsum().shift(fill_value=0)

# Устанавливаем значение 0 для каждой единицы в 'sum'
df.loc[df['y_test'] == 1, 'sum'] = 0

# Добавляем столбец 'diff'
df['diff'] = 0.0

# Для каждого нуля в 'y_test' рассчитываем значение 'diff'
for index, row in df.iterrows():
    if row['y_test'] == 0:
        # Получаем количество единиц с таким же значением y_proba
        count_ones = df[df['y_proba'] == row['y_proba']]['y_test'].sum()
        # Если есть единицы, добавляем n * 0.5
        if count_ones > 0:
            df.at[index, 'diff'] = count_ones * 0.5

df['result'] = df['sum'] - df['diff']

roc_auc = df['result'].sum() / (pos * neg)
print(roc_auc)
print(roc_auc_score(y_test, y_proba))
