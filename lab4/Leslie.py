import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def deathrate(i, t, DeathRate):
    if i <= 5 or i >= 50:
        dr = DeathRate[i] * (2 - (t - 2010) * 1e-3)
    else:
        dr = DeathRate[i]
    return dr


def A(t, m, DeathRate):
    At = np.zeros((m, m))
    for i in range(m-1):
        At[i+1, i] = 1 - deathrate(i, t, DeathRate)
    return At


def B(t, i1, i2, m, DeathRate, SexRatio):
    Bt = np.zeros((m, m))
    for i in range(i1, i2):
        h = pow(i - i1, 4) * np.exp(-(i - i1) / 2) / 767.886
        ki = 100 / (SexRatio[i] + 100)
        res = (1 - deathrate(1, t, DeathRate))* h * ki
        Bt[0, i] = (1 - deathrate(1, t, DeathRate)) * h * ki
    # print(Bt[0])
    return Bt


data_population = pd.read_csv('./assets/第六次人口普查.csv', header=None, usecols=[1])
data_deathrate = pd.read_csv('./assets/第六次人口普查-死亡.csv', header=None, usecols=[1])
data_sexratio = pd.read_csv('./assets/第六次人口普查-性别.csv', header=None, usecols=[0])

data_population, data_deathrate, data_sexratio = np.array(data_population), np.array(data_deathrate), \
                                                 np.array(data_sexratio)

AgePopulation = data_population[:, 0]
DeathRate = data_deathrate[:, ] / 1000
SexRatio = data_sexratio[:, 0]

data_population1 = pd.read_csv('./assets/人口总数表.csv', encoding='gbk', usecols=[1])
data_population1 = np.array(data_population1)

population = data_population1[55:73, 0]
popu2 = population[6:17]
year_range = range(2010, 2021)
m = 100
years = 31
x = np.zeros((100, years))
x[:,0] = data_population[:,0]
year = range(2010, 2010+years)
beta = np.arange(1.5, 2.7, 0.3)
popu1 = np.zeros((4, years))
for num in range(4):
    for i in range(1, years):
        A1 = A(year[i-1], m, DeathRate)
        B1 = B(year[i-1], 15, 49, m, DeathRate, SexRatio)
        res = A1 @ x[:, i - 1] + beta[num] * (B1 @ x[:, i-1])
        x[:, i] = A1 @ x[:, i - 1] + beta[num] * (B1 @ x[:, i-1])
    popu = np.sum(x, axis=0)
    popu1[num, :] = popu
popu3 = popu1[3, :]
plot1 = plt.plot(year, popu1[0, :], label="beta=1.5")
plot2 = plt.plot(year, popu1[1, :], label="beta=1.8")
plot3 = plt.plot(year, popu1[2, :], label="beta=2.1")
plot4 = plt.plot(year, popu1[3, :], label="beta=2.4")
plot0 = plt.plot(year_range, popu2*10**4, 's', label='real value')
plt.xlabel('year')
plt.ylabel('population')
plt.title('Leslie model result')
plt.legend(loc='upper right')
plt.show()