import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./assets/人口总数表.csv', encoding='gbk', usecols=[1],)
data = data[20:]
t = range(len(data))
data = np.array(data)
# Logistic模型
p00 = data[0, 0]
r = np.polyfit(data[1:, 0], np.diff(data[:, 0])/data[1:, 0], 1)
p0m = -r[1]/r[0]
r0 = r[1]
f0 = lambda x: p0m/(1+(p0m/p00-1)*np.exp(-r0*x))
# print(f0(t))
err = np.linalg.norm(f0(t)-data)
t_pre0 = range(len(data)+10)
y_pre0 = f0(t_pre0)

# 改良的logistic模型
p0 = data[0, 0]
r_x = np.polyfit(data[1:, 0], np.diff(data[:, 0])/data[1:, 0], 2)
pm = -r_x[1]/r_x[0]

y = -np.log((pm*p0-data[:, 0]*p0)/(pm*data[:, 0]-data[:, 0]*p0))
k = np.polyfit(t, y, 3)

f = lambda x: pm/(1+np.exp(-np.polyval(k, x))*(pm/p0-1))

err = np.linalg.norm(f(t)-data[:, 0], ord=2)
y = f(t)
t_pre = range(len(data)+10)
y_pre = f(t_pre)

x, error = [], []
for i in range(len(t)):
    erroring = abs(f(t)[i]-data[:, 0][i])/data[:,0][i]
    error.append(erroring)
print("模型对于人口数据的拟合值为（1969-2021）：\n", f(t))
print("拟合值和真实值的相对误差为：\n", error)
t, t_pre = np.array(t), np.array(t_pre)
plot1 = plt.plot(t+1969, y, label="logistic pro fitted", marker='*')
plot2 = plt.plot(t+1969, data, 'r', label='true value')
plot3 = plt.plot(t_pre[53:]+1969, y_pre[53:], 's', label='logistic pro predict')
plot4 = plt.plot(t+1969, y_pre0[:53], 's', label="logistic fitted")
plot5 = plt.plot(t_pre[53:]+1969, y_pre0[53:], 's', label='logistic predict')
plt.xlabel('time')
plt.ylabel('population')
plt.legend(loc=0)  # 指定legend的位置右下角
plt.show()



