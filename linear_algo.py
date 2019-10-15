#coding=utf-8
import numpy as np
from scipy.optimize import leastsq, linprog
import  matplotlib.pylab as plt
import sys


# 线性规划问题
"""
max: z = 2*x1 + 3*x2 - 5*x3
s.t. x1 + x2 + x3 = 7
     2*x1 - 5*x2 + x3 >= 10
     x1 + 3*x2 + x3 <= 12
     x1, x2, x3 >= 0

=>
min: -z = -2*x1 -3*x2 + 5*x3
s.t. -2*x1+5*x2-x3<=-10
     x1 + 3*x2 + x3 <=12
     x1 + x2 + x3 = 7
     x1, x2, x3 >= 0
c = [-2, -3, 5]
Aup = [[-2, 5, -1], [1, 3, 1]]
Bup = [-10, 12]
Aeq = [[1, 1, 1]]
Beq = [7]
x1_bound, x2_bound, x3_bound = (0, None), (0, None), (0, None)
bounds = [x1_bound, x2_bound, x3_bound]
"""

c=np.array([2,3,-5])
A_ub=np.array([[-2,5,-1],[1,3,1]])
B_ub=np.array([-10,12])
A_eq=np.array([[1,1,1]])
B_eq=np.array([7])
x1=(0,None)
x2=(0,None)
x3=(0,None)
res=linprog(-c,A_ub,B_ub,A_eq,B_eq,bounds=(x1,x2,x3))
print(res)



# 最小二乘问题(min-square 问题)
ti = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
yi = np.array([8, 11, 15, 19, 22, 23, 22, 19, 15, 11])

# matrix

def func(a, b, c, d, e, x):
    return a*(x**4)+b*x**3 +c*x**2+d*x + e


def loss_func(param, x, y): # x,y 为训练数据中的一个元素，可以为矩阵
    a, b, c, d, e = param
    loss = func(a, b, c, d, e, x)-y+0.001*(abs(a)+abs(b)+abs(c)+abs(d)+abs(e))
    return loss

initial_param = np.array([0, 0, 0, 0, 0])
s = leastsq(loss_func, initial_param, args=(ti, yi))
print(s[0])


a, b, c, d, e = s[0]
y_hat = func(a, b, c, d, e, ti)
# draw
plt.scatter(ti, yi, facecolor='red')

plt.plot(ti, y_hat)
plt.show()


