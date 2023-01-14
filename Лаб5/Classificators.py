import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# type - тип классификатора
def classifier(x_train, x_test, y_train, y_test, type):
    # Создание модели
    match type:
        case 'knn':
            clf = KNeighborsClassifier(n_neighbors=5)
            label = 'К ближайших соседей'
        case 'nb':
            clf = GaussianNB()
            label = 'Наивный Байесовский классификатор'
        case 'rf':
            clf = RandomForestClassifier(n_estimators=100)
            label = 'Случайный лес'
    # обучение модели на обучающих данных
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    print(label)
    print(classification_report(y_test, y_prediction))

    # Визуализация результата обучения модели
    spectral_classes = pd.DataFrame(y_prediction).value_counts()
    pies(spectral_classes, label)
    return accuracy_score(y_prediction, y_test)


# Круговая диаграмма результатов работы модели
def pies(source, title):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.pie(source, labels=source.keys(), autopct='%1.1f%%')
    ax1.axis('equal')
    plt.show()


# Поменять местами два столбца в DataFrame
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
