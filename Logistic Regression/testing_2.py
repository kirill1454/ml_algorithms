import pandas as pd

# Создаем DataFrame
data = {
    'y_proba': [0.96, 0.88, 0.77, 0.6, 0.6, 0.6],
    'y_test': [1, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Создаем новый столбец 'sum', который будет содержать накопительную сумму единиц
df['sum'] = df['y_test'].cumsum().shift(fill_value=0)

# Устанавливаем значение 0 для каждой единицы в 'sum'
df.loc[df['y_test'] == 1, 'sum'] = 0

# Инициализируем новый столбец diff
df['diff'] = 0.0

# Проходим по всем строкам DataFrame
for i in range(len(df)):
    if df.loc[i, 'y_test'] == 0:  # Если y_test равно 0
        # Считаем количество совпадений y_proba для y_test = 1
        count = df.loc[df['y_test'] == 1, 'y_proba'].value_counts().get(df.loc[i, 'y_proba'], 0)
        # Если совпадения найдены, добавляем n * 0.5 в diff
        df.loc[i, 'diff'] = count * 0.5

df['result'] = df['sum'] - df['diff']

# Выводим результат
print(df)
print(sum(df['result']))