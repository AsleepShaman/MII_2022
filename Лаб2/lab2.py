#Лабораторная работа №2
#Кучина Анна, ИСТбд-42
#Вариант 12

import numpy as np
import matplotlib.pyplot as plt


def is_prime(k):
    for i in range(2, int(k/2)+1):
         if (k % i) == 0:
            return False
    else:
        return True

def primes(matrix, matrix_length):
    countPrime = 0
    for i in range(matrix_length):
        for j in range(matrix_length):
            if j%2 == 0:
                if is_prime(matrix[i][j])==True:
                    countPrime =+ 1
    return countPrime

def funk_summ(matrix, matrix_length):
    countSum = 0
    for i in range(matrix_length):
        if i % 2 != 0:
            for j in range(matrix_length):
                countSum = countSum + matrix[i][j]
    return countSum


# K = int(input("Введите число K: "))
# N = int(input('Введите число N(больше 3 и кратное 2): '))
K = 4
N = 4

if N < 4 or (N % 2 != 0):
    print('N не корректна!')
    exit()

N = N // 2

b = np.random.randint(-10, 10, (N, N))
c = np.random.randint(-10, 10, (N, N))
d = np.random.randint(-10, 10, (N, N))
e = np.random.randint(-10, 10, (N, N))

A = np.vstack([np.hstack([e, b]), np.hstack([d, c])])
print('Матрица A: \n')
print(A, '\n')

countSimple = primes(b, N)
countSumm = funk_summ(b, N)

print("Количество простых чисел в четных столбцах: ", countSimple)
print("Сумма чисел в четных строках: ", countSumm)


if countSimple > countSumm:
    ef = np.flip(e, 1)
    bf = np.flip(b, 1)
    F = np.vstack([np.hstack([bf, ef]), np.hstack([d, c])])
else:
    F = np.vstack([np.hstack([c, b]), np.hstack([d, e])])

print('Матрица F: \n')
print(F, '\n')



if np.linalg.det(A) > np.diagonal(F).sum():
    result = np.linalg.inv(A) * np.transpose(A) - K * np.linalg.inv(F)
else:
    result = (np.linalg.inv(A) + np.tril(A) - np.linalg.inv(F)) * K

print('Результ вычислений: \n')
print(result)

figFirst = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(F[:N, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 2)
plt.imshow(F[:N, N:], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 3)
plt.imshow(F[N:, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 4)
plt.imshow(F[N:, N:], cmap='rainbow', interpolation='bilinear')
plt.show()

figSecond = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(F[:N, :N])
plt.subplot(2, 2, 2)
plt.plot(F[:N, N:])
plt.subplot(2, 2, 3)
plt.plot(F[N:, :N])
plt.subplot(2, 2, 4)
plt.plot(F[N:, N:])
plt.show()
