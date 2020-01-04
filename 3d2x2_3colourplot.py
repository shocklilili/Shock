
import numpy as np
from pandas import DataFrame
'''
mount = 10000   # amount of sample
rand_nps = []

for k in range(mount):
    # create matrix container
    arr = [0] * 5
    a, b, c = [np.random.randint(-100, 100) for i in range(3)]
    rand_np = [[a, b], [b, c]]
    for i in range(2):
        for j in range(2):
            print("i = {} and j={}".format(i, j))
            index = i * 2 + j
            print(index)
            arr[index] = rand_np[i][j]
    eigenvl, _ = np.linalg.eig(rand_np)
    s = 0
    #data1 = DataFrame(rand_np)
    #data1.to_csv('train.csv')
    for e in eigenvl:
        if e > 0:
            s += 1
    if s > 1:
        arr[-1] = 2
    elif s > 0:
        arr[-1] = 1
    else:
        arr[-1] = 0
    print(arr)
    rand_nps.append(arr)

nps = np.array(rand_nps).reshape(mount, -1)
print(nps)
data1 = DataFrame(nps)
data1.to_csv('abbc_train.csv')
'''

#MAIN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#data = data1

#x, y, z, q = data[0], data[1], data[3], data[4]
#print('x=',x,'y=',y)
ax = plt.subplot(111, projection='3d')  # create a 3d plot project
#  separate data to 3 part, classify by colour

df = pd.read_csv('abbc_train.csv', encoding='gbk')
df_0 = df[df['r'] == 0].copy()
#print(df_0)
#lf = list(df_0)
#print('df_0 = ', lf[0])
x0 = pd.to_numeric(df_0['a'])
y0 = pd.to_numeric(df_0['b'])
z0 = pd.to_numeric(df_0['c'])
#print('x0 = ', x0)
df_1 = df[df['r'] == 1].copy()
x1 = pd.to_numeric(df_1['a'])
y1 = pd.to_numeric(df_1['b'])
z1 = pd.to_numeric(df_1['c'])
df_2 = df[df['r'] == 2].copy()
x2 = pd.to_numeric(df_2['a'])
y2 = pd.to_numeric(df_2['b'])
z2 = pd.to_numeric(df_2['c'])

#fig = plt.figure()
#ax = plt.subplot()
ax.scatter(x0, y0, z0, c='yellow')
ax.scatter(x1, y1, z1, c='r')
ax.scatter(x2, y2, z2, c='b')

ax.set_zlabel('Z')  # coordinate axis
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

'''
if c[1]
df[df["F"]] == 0
x0, y0, z0 = c0[0], c0[1], c0[2]
print('c0 = ',x0,y0,z0)
c1 = c.loc[data[4] == '1']
x1, y1, z1 = c1[0], c1[1], c1[2]


ax.scatter(x0, y0, z0, c='y')  # plot dot
ax.scatter(x1, y1, z1, c='r')
ax.scatter(x2, y2, z2, c='g')

ax.set_zlabel('Z')  # coordinate axis
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

'''
