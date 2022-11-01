import csv
import random
import string
import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Массив заголовков
Titles = ['Id', 'Name', 'Sex', 'BirthDate', 'Start working', 'Store', 'Poste', 'Salary', 'Works']

# Массивы с данными для рандомизации
# Имена
Names = ['Brown', 'Smith', 'Johnson', 'Williams', 'Simons', 'Davis', 'Wilson', 'Taylor', 'Moore',
         'Clark', 'Robinson']
# Пол
Sex = ['Male', 'Female']
# Отдел
Stores = ['Web', 'System developing', 'Mobile Developing', 'Support']
# Должность
Posts = ['TeamLead', 'database architect', 'network administrator', 'system engineer',
         'android developer', 'ios developer', 'software tester', 'designer']


# Мода распределения
def mode(values):
    dict = {}
    for elem in values:
        if elem in dict:
            dict[elem] += 1
        else:
            dict[elem] = 1
    v = list(dict.values())
    k = list(dict.keys())

    return k[v.index(max(v))]


#Функция для расчета статистических характеристик с помощью библиотеки numpy
def np_statistics(column, csv_list, headers):
    print('Для столбца ' + headers[column] + '\n')
    stat = []

    for row in csv_list:
        x = int(row[column])
        stat.append(x)

    statist = numpy.array(stat)

    print('Минимальное значение: ' + str(numpy.min(statist)))
    print('Максимальное значение: ' + str(numpy.max(statist)))
    print('Математическое ожидание: ' + str(numpy.mean(statist)))
    print('Стандартное отклонение: ' + str(numpy.std(statist)))
    print('Дисперсия: ' + str(numpy.var(statist)))
    print('Медиана: ' + str(numpy.median(statist)))
    print('Мода: ' + str(mode(statist)) + '\n')


#Функция для расчета статистических данных с помощью pandas
def pandas_statistics(dataframe, column):
    print('Для ' + column + ':\n')
    print('Минимальное значение: ' + str(dataframe[column].min()))
    print('Максимальное значение: ' + str(dataframe[column].max()))
    print('Математическое ожидание: ' + str(dataframe[column].mean()))
    print('Стандартное отклонение: ' + str(dataframe[column].std()))
    print('Дисперсия: ' + str(dataframe[column].var()))
    print('Медиана: ' + str(dataframe[column].median()))
    print('Мода: ' + str(dataframe[column].mode()) + '\n')


# Заполнение файла данными
with open('sw_data_new.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\r")
    writer.writerow(Titles)
    for i in range(1, 1500):
        name = random.choice(Names) + " " + random.choice(string.ascii_uppercase) + "."
        sex = random.choice(Sex)
        birthdate = random.randrange(1967, 2002, 1)
        stage = random.randrange(1, 15, 1) if birthdate > 21 else random.randrange(1, 3, 1)
        store = random.choice(Stores)
        poste = random.choice(Posts)
        payment = random.randrange(90000, 400000, 5000)
        works = random.randrange(0, 20, 1)
        row = [i, name, sex, birthdate, stage, store, poste, payment, works]
        writer.writerow(row)

# чтение данных из csv файла
my_list = list()

with open('sw_data_new.csv') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        my_list.append(row)

#Расчет статистических характеристик с пом-ю библиотеки numpy
np_statistics(3, my_list, headers)
np_statistics(7, my_list, headers)
np_statistics(8, my_list, headers)

# чтение в DataFrame
df = pandas.read_csv('sw_data_new.csv', header=0, index_col=0)

#Расчет статистических характеристик с пом-ю библиотеки pandas
pandas_statistics(df, 'BirthDate')
pandas_statistics(df, 'Salary')
pandas_statistics(df, 'Works')

#Построение графика зависимостей
data = [df["Sex"].value_counts()["Male"], df["Sex"].value_counts()["Female"]]
plt.pie(data, labels=["Male", "Female"])
plt.title("Круговая диаграмма полов в компании")
plt.ylabel("")
plt.show()

#Построение круговой диаграммы
graf1 = df['Store'].hist()
plt.xlabel('department')
plt.ylabel('number of employees')
plt.xticks(rotation=90)
plt.title("Количество сотрудников в отделах")
plt.show()

#Построение графика зависимости другого типа
plt.figure(figsize=(16, 10), dpi=80)
plt.plot_date(df["Start working"], df["Salary"])
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.ylabel('salary')
plt.xlabel('dates')
plt.title("Изменение зарплаты со временем")
plt.show()
