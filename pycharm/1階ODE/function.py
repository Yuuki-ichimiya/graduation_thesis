from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # HHL algorithmに必要な関数
from copy import copy

np.set_printoptions(linewidth=200, precision=20)

# 各パラメータ
number = 3
scale_fac = 1.0
reg_nbit_list = np.array([9, 10, 11])  # 位相推定に使うレジスタの数(10,11,12)

if number == 0:
    # dy/dx = -2(y-1)
    h = 0.05  # 刻み幅
    n = 200  # 考える区間[x_first, x_first + h*(n-1)]
    order = 1
    x_list = np.array([h * i for i in range(n)])
    y_exact = np.exp(-2 * x_list) + 1
    y_first = copy(y_exact)


elif number == 1:
    # dy/dx = -2x(y-1)
    h = 0.05  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    order = 1
    x_list = np.array([h * i for i in range(n)])
    y_exact = np.exp(-x_list ** 2) + 1
    y_first = copy(y_exact)


elif number == 2:
    # dy/dx = -2x(y-2)^2-4(y-2)
    h = 0.01  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    order = 2
    x_list = np.array([h * i for i in range(n)])
    y_exact = 2 - 8 / (7 * np.exp(4 * x_list) + 4 * x_list + 1)
    y_first = copy(y_exact)


elif number == 3:
    h = 0.01  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    order = 3
    x_list = np.array([h * i for i in range(n)])
    y_exact = 2 + 1 / np.sqrt(4 + 2 * x_list ** 2)  # dy/dx = -2x(y-2)^3
    y_first = copy(y_exact)


def function(x):
    if number == 0:
        return np.array([2, -2])
    elif number == 1:
        return np.array([2 * x, -2 * x])

    elif number == 2:
        return np.array([-8 * x + 8, 8 * x - 4, -2 * x])

    elif number == 3:
        return np.array([16 * x, -24 * x, 12 * x, -2 * x])  # dy/dx = -2x(y-2)^3


depth = int(np.floor(np.log2(n)))
