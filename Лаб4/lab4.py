# Лабораторная работа 4
# Кучина Анна, ИСТбд-42
# Метод k-nn, Парзеновское окно, веса классов, библиотека sklearn

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Расчет Евклидового расстояния
def distance(x1, y1, x2, y2):
    return (abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2) ** 0.5


# Метод knn без использования библиотек с парзеновским окном
def knn(dataset, teach, pars_window):
    data = np.array(dataset)
    test = len(data) - teach
    success = 0     # количество правильно классифированных объектов
    classes = [0] * test       # классы тестовой выборки
    new_dist = np.zeros((test, teach))      # матрица евклидовых расстояний для тестовой выборки

    # заполнение матрицы расстояний
    for i in range(test):
        for j in range(teach):
            dist = distance(int(data[teach + i][1]), int(data[teach + i][2]), int(data[j + 1][1]), int(data[j + 1][2]))
            # если посчитанное расстояние больше значения парзеновского окна - обесценить его (присвоить большое число)
            new_dist[i][j] = dist if dist < pars_window else 1000

    # Для каждого элемента тестовой выборки
    for i in range(test):
        print(str(i) + '. Классификация ', data[teach + i][0])
        # веса каждого класса, их количество - количество различных классов в исходном наборе данных
        weights = [0] * dataset.iloc[:]['Класс'].nunique()
        neighbor = np.sum(new_dist[i] != 1000)      # количесество элементов внутри парзеновского окна
                                                    # для каждого объекта тестовой выборки
        # для каждого значимого соседа объекта тестовой выборки
        for j in range(neighbor + 1):
            ind_min = new_dist[i].argmin()
            # увеличисть вес класса, совпадающего с индексом минимального элемента обучающей выборки
            # на величину (k-j+1)/k
            weights[int(data[ind_min + 1][3])] += ((neighbor - j + 1) / neighbor)
            # обесценивание текущего минимального элемента
            new_dist[i][ind_min] = 1000
            print('индекс соседа =', ind_min, 'сосед -', data[ind_min + 1][0])
        # класс текущего объекта - это класс с максимальным весом
        classes[i] = np.array(weights).argmax()
        print('Полученный элемента =', classes[i], 'Реальный класс элемента =', data[teach + i][3])
        if int(classes[i]) != int(data[teach + i][3]):
            print('не совпал')
        else:
            print('Совпал')
            success += 1

    print(classes)
    print('Количество совпадений:', str(success))
    return classes


# Метод knn с использованием библиотеки sklearn
def knn_sklearn(dataset):
    # создание наборов параметров и классов
    x = dataset.iloc[:, 1:3].values
    y = dataset.iloc[:, 3].values

    # обучающая и тестовая выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=False, stratify=None)

    # метод knn из библиотеки sklearn с 4-мя соседями
    classifier = KNeighborsClassifier(n_neighbors=4)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # обучение
    classifier.fit(x_train, y_train)
    # проверка на тестовой выборке
    y_prediction = classifier.predict(x_test)
    # матрицы для представления результатов
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))


# Визуализация набора данных
def graphic(dataset, window):
    colours = {'0': 'orange', '1': 'blue', '2': 'green', '3': 'red'}
    sweet = dataset['Сладость']
    crun = dataset['Хруст']
    col_list = [colours[str(i)] for i in dataset['Класс']]
    pylab.subplot(2, 1, window)
    plt.scatter(sweet, crun, c=col_list)
    plt.title('График входных данных')
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')


# набор данных для трех классов
food = [["Продукт", "Сладость", "Хруст", "Класс"],
        ['Яблоко', '7', '7', '0'],
        ['Салат', '2', '5', '1'],
        ['Бекон', '1', '2', '2'],
        ['Орехи', '1', '5', '2'],
        ['Рыба', '1', '1', '2'],
        ['Сыр', '2', '1', '2'],
        ['Банан', '9', '1', '0'],
        ['Морковь', '2', '8', '1'],
        ['Виноград', '8', '1', '0'],
        ['Апельсин', '6', '1', '0'],
        # test set of 5 (row 11-16)
        ['Клубника', '9', '1', '0'],
        ['Капуста', '3', '7', '1'],
        ['Шашлык', '3', '1', '2'],
        ['Груша', '5', '3', '0'],
        ['Сельдерей', '1', '7', '1']]

# набор данных для четырех классов
food_upgrade = [["Продукт", "Сладость", "Хруст", "Класс"],
                ['Яблоко', '7', '4', '0'],
                ['Карамель', '10', '9', '3'],
                ['Салат', '2', '5', '1'],
                ['Маскарпоне', '10', '8', '3'],
                ['Бекон', '1', '2', '2'],
                ['Орехи', '1', '5', '2'],
                ['Леденец', '8', '10', '3'],
                ['Рыба', '1', '1', '2'],
                ['Сыр', '2', '1', '2'],
                ['Хворост', '6', '9', '3'],
                ['Банан', '9', '1', '0'],
                ['Морковь', '2', '8', '1'],
                ['Драже', '6', '7', '3'],
                ['Виноград', '8', '1', '0'],
                # test set of 6 (row 13-21)
                ['Апельсин', '6', '1', '0'],
                ['Клубника', '9', '1', '0'],
                ['Шоколад', '7', '6', '3'],
                ['Капуста', '3', '7', '1'],
                ['Шашлык', '3', '1', '2'],
                ['Груша', '6', '3', '0'],
                ['Сельдерей', '1', '7', '1']]

# Запись данных в csv файл
with open('food_csv.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f, lineterminator="\r")
    for row in food:
        writer.writerow(row)
with open('food_upgrade_csv.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f, lineterminator="\r")
    for row in food_upgrade:
        writer.writerow(row)

# количество элементов обучающей выборки в наборе и значение парзеновского окна
teach_number = 10
pars = 4

# Чтение данных из csv файла
dataset_upgrade = pd.read_csv('food_upgrade_csv.csv')
dataset_1 = pd.read_csv('food_csv.csv')
start_data = dataset_1[:teach_number]['Класс']  # классы обучающей выборки

# Метод knn без библиотек для первого набора данных с тремя классами
s1 = pd.Series(knn(dataset_1, teach_number, pars))
start_data = pd.concat([start_data, s1])

# Визуализация исходного набора данных и обучающей+тестовой выборки после применения метода
colours = {'0': 'orange', '1': 'blue', '2': 'green'}
graphic(dataset_1, 1)
colour_list = [colours[str(i)] for i in start_data]
graphic(dataset_1, 2)
plt.show()

# Метод knn без библиотек для второго набора данных с четыремя классами
teach_number = 14
s1 = pd.Series(knn(dataset_upgrade, teach_number, pars))
start_data = pd.concat([start_data, s1])

# Визуализация исходного набора данных и обучающей+тестовой выборки после применения метода
graphic(dataset_upgrade, 1)
graphic(dataset_upgrade, 2)
plt.show()

# Метод knn с библиотекой sklearn для двух наборов данных
knn_sklearn(dataset_1)
knn_sklearn(dataset_upgrade)
