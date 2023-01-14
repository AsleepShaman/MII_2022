# Лабораторная работа 5
# Кучина Анна, ИСТбд-42
# Классификация, КНН, наивный Байесовский классификатор, случайный лес, библиотека sklearn

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import Classificators as cl

stars_dataset = pd.read_csv('stars_classification.csv', delimiter=',')
stars_dataset.drop('Star color', axis=1, inplace=True)
stars_dataset = cl.swap_columns(stars_dataset, 'Star type', 'Spectral Class')
Spectrals = {'M': 6, 'A': 2, 'O': 0, 'F': 3, 'G': 4, 'B': 1, 'K': 5}
stars_dataset = stars_dataset.replace({'Spectral Class': Spectrals})

x = stars_dataset.iloc[:, 1:4].values
y = stars_dataset.iloc[:, 5].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, shuffle=True, stratify=None)

# Визуализация значимости каждого признака
label = stars_dataset['Star type']
stars_dataset.drop('Star type', axis=1, inplace=True)
continuous_features = set(stars_dataset.columns)
# Масштабирование признаков
scaler = MinMaxScaler()
df_norm = stars_dataset.copy()
df_norm[list(continuous_features)] = scaler.fit_transform(stars_dataset[list(continuous_features)])

# Построение графика визуализации
clf = RandomForestClassifier()
clf.fit(df_norm, label)
plt.figure(figsize=(12, 12))
plt.bar(df_norm.columns, clf.feature_importances_, color='green')
plt.title("Значимость признаков с использованием случайного леса")
plt.yticks(fontsize=6)
plt.show()

print(stars_dataset.info())

knn_ac = cl.classifier(x_train, x_test, y_train, y_test, 'knn')
nb_ac = cl.classifier(x_train, x_test, y_train, y_test, 'nb')
rf_ac = cl.classifier(x_train, x_test, y_train, y_test, 'rf')

# Визуализация точности алгоритмов обучения
plt.plot(['KNN', 'NB', 'RF'], [knn_ac, nb_ac, rf_ac])
plt.title("Точность алгоритмов классификации")
plt.show()
